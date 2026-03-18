"""
Recent form calculator for CBB Totals Model.
Fetches and aggregates per-game data to compute rolling statistics.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Sentinel for missing data
MISSING = -1.0

# League average PPP for possession estimation
_LEAGUE_PPP = 1.05


class RecentFormCalculator:
    """
    Computes rolling statistics and form metrics for a team.

    Usage::

        calc = RecentFormCalculator()
        games = calc.get_recent_games(team_id, n=10, before_date="2024-02-15", db=db)
        stats = calc.compute_rolling_stats(games, window=5)
    """

    # ── Fetching games ────────────────────────────────────────────────────

    def get_recent_games(
        self,
        team_id: str,
        n: int,
        before_date: str,
        db_manager,
    ) -> list[dict]:
        """
        Fetch the last N completed games for a team before a given date.

        Args:
            team_id:    Team identifier.
            n:          Maximum number of games to return.
            before_date: Date string (YYYY-MM-DD); only games strictly before
                         this date are included.
            db_manager: DatabaseManager instance.

        Returns:
            List of game dicts, newest first.  May be fewer than N if not
            enough completed games are available.
        """
        if db_manager is None:
            logger.warning("get_recent_games: db_manager is None.")
            return []

        try:
            games = db_manager.get_recent_games(team_id, n, before_date)
        except Exception as exc:
            logger.warning(f"Could not fetch recent games for {team_id}: {exc}")
            return []

        # Enrich each game: add 'score' / 'opp_score' from team's perspective
        enriched = []
        for g in games:
            enriched.append(_orient_game(g, team_id))

        return enriched

    def get_rest_days(
        self,
        team_id: str,
        game_date: str,
        db_manager,
    ) -> int:
        """
        Return the number of days between the team's last game and game_date.

        Args:
            team_id:    Team identifier.
            game_date:  Date of the upcoming game (YYYY-MM-DD).
            db_manager: DatabaseManager instance.

        Returns:
            Days since last game (0 if no prior game found, max capped at 30).
        """
        last_date_str = self.get_last_game_date(team_id, game_date, db_manager)
        if not last_date_str:
            return 0

        try:
            last = date.fromisoformat(last_date_str)
            upcoming = date.fromisoformat(game_date)
            delta = (upcoming - last).days
            return min(max(delta, 0), 30)
        except ValueError:
            return 0

    def get_last_game_date(
        self,
        team_id: str,
        before_date: str,
        db_manager,
    ) -> Optional[str]:
        """
        Return the date string of the team's most recent game before before_date.

        Returns None if no prior game found.
        """
        if db_manager is None:
            return None
        try:
            games = db_manager.get_recent_games(team_id, 1, before_date)
            if games:
                return games[0].get("date")
        except Exception as exc:
            logger.debug(f"get_last_game_date error for {team_id}: {exc}")
        return None

    # ── Rolling statistics ────────────────────────────────────────────────

    def compute_rolling_stats(
        self,
        games: list[dict],
        window: int,
    ) -> dict:
        """
        Compute rolling averages over the last `window` games.

        Args:
            games:  List of game dicts (newest first, already oriented for team).
            window: Number of games to include.

        Returns:
            Dict with keys: avg_pts, avg_opp_pts, avg_total,
            avg_pace, avg_off_eff, avg_def_eff, games_counted.
        """
        if not games:
            return _empty_rolling_stats()

        recent = games[:window]
        n = len(recent)

        pts_list = [g["score"] for g in recent if g.get("score") is not None]
        opp_pts_list = [g["opp_score"] for g in recent if g.get("opp_score") is not None]
        total_list = [g["total"] for g in recent if g.get("total") is not None]
        poss_list = [g["possessions"] for g in recent if g.get("possessions") is not None]

        avg_pts = _safe_mean(pts_list)
        avg_opp_pts = _safe_mean(opp_pts_list)
        avg_total = _safe_mean(total_list)

        if poss_list:
            avg_pace = _safe_mean(poss_list)
        elif total_list:
            # Rough estimate: total pts / (2 * PPP) ≈ possessions
            estimated = [t / (2 * _LEAGUE_PPP) for t in total_list]
            avg_pace = _safe_mean(estimated)
        else:
            avg_pace = MISSING

        # Efficiencies per 100 possessions
        if avg_pace and avg_pace > 0 and avg_pts != MISSING:
            avg_off_eff = (avg_pts / avg_pace) * 100.0
        else:
            avg_off_eff = MISSING

        if avg_pace and avg_pace > 0 and avg_opp_pts != MISSING:
            avg_def_eff = (avg_opp_pts / avg_pace) * 100.0
        else:
            avg_def_eff = MISSING

        return {
            "avg_pts": avg_pts,
            "avg_opp_pts": avg_opp_pts,
            "avg_total": avg_total,
            "avg_pace": avg_pace,
            "avg_off_eff": avg_off_eff,
            "avg_def_eff": avg_def_eff,
            "games_counted": n,
        }

    def compute_rolling_pace(
        self,
        games: list[dict],
        window: int,
    ) -> float:
        """
        Compute average pace (possessions per game) over the last N games.

        Returns MISSING if insufficient data.
        """
        stats = self.compute_rolling_stats(games, window)
        return stats.get("avg_pace", MISSING)

    def compute_form_trend(
        self,
        games: list[dict],
    ) -> float:
        """
        Compute scoring form trend: average points in last 3 games vs
        average points in all available games (or last 10).

        Positive value → team scoring more recently than season average.
        Negative value → team scoring less recently.

        Returns MISSING if fewer than 3 games available.
        """
        if not games or len(games) < 3:
            return MISSING

        recent_3 = games[:3]
        baseline = games[:10] if len(games) >= 10 else games

        pts_recent = [g["score"] for g in recent_3 if g.get("score") is not None]
        pts_baseline = [g["score"] for g in baseline if g.get("score") is not None]

        avg_recent = _safe_mean(pts_recent)
        avg_baseline = _safe_mean(pts_baseline)

        if avg_recent == MISSING or avg_baseline == MISSING or avg_baseline == 0:
            return MISSING

        return avg_recent - avg_baseline

    def compute_all_windows(
        self,
        games: list[dict],
        windows: list[int] = None,
        side: str = "home",
    ) -> dict:
        """
        Compute rolling stats for multiple window sizes at once.

        Args:
            games:   List of game dicts (newest first).
            windows: List of window sizes (default [3, 5, 10]).
            side:    'home' or 'away' (used in result key naming).

        Returns:
            Flat dict with all window stats prefixed by '{side}_last_{N}_'.
        """
        if windows is None:
            windows = [3, 5, 10]

        result = {}
        for w in windows:
            stats = self.compute_rolling_stats(games, w)
            for k, v in stats.items():
                if k == "games_counted":
                    continue
                result[f"{side}_last_{w}_{k.replace('avg_', '')}"] = v

        # Form trend
        trend = self.compute_form_trend(games)
        result[f"{side}_form_trend"] = trend

        return result

    def get_streak_info(self, games: list[dict]) -> dict:
        """
        Compute win/loss streak and over/under streak for a team.

        Args:
            games: List of game dicts (newest first).

        Returns:
            Dict with: win_streak, loss_streak, over_streak, under_streak,
            current_streak_type ('W' | 'L' | 'N'), current_streak_len.
        """
        if not games:
            return {
                "win_streak": 0, "loss_streak": 0,
                "over_streak": 0, "under_streak": 0,
                "current_streak_type": "N", "current_streak_len": 0,
            }

        # Determine current W/L streak from most recent game
        current_result = None
        current_len = 0
        win_streak = 0
        loss_streak = 0

        for g in games:
            won = g.get("winner")
            if won is None:
                # Try to infer from score
                score = g.get("score")
                opp_score = g.get("opp_score")
                if score is not None and opp_score is not None:
                    won = score > opp_score
                else:
                    continue

            this_result = "W" if won else "L"
            if current_result is None:
                current_result = this_result
                current_len = 1
            elif this_result == current_result:
                current_len += 1
            else:
                break

        win_streak = current_len if current_result == "W" else 0
        loss_streak = current_len if current_result == "L" else 0

        return {
            "win_streak": win_streak,
            "loss_streak": loss_streak,
            "over_streak": 0,    # Would need market line to compute
            "under_streak": 0,
            "current_streak_type": current_result or "N",
            "current_streak_len": current_len,
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_mean(values: list) -> float:
    """Mean of a list; returns MISSING if list is empty."""
    valid = [float(v) for v in values if v is not None]
    if not valid:
        return MISSING
    return sum(valid) / len(valid)


def _empty_rolling_stats() -> dict:
    return {
        "avg_pts": MISSING,
        "avg_opp_pts": MISSING,
        "avg_total": MISSING,
        "avg_pace": MISSING,
        "avg_off_eff": MISSING,
        "avg_def_eff": MISSING,
        "games_counted": 0,
    }


def _orient_game(game: dict, team_id: str) -> dict:
    """
    Reframe a generic game dict so that 'score' / 'opp_score' are from
    the perspective of `team_id` (regardless of whether they were home/away).

    Adds keys: score, opp_score, total, home (bool), winner (bool | None).
    """
    home_id = game.get("home_team_id", "")
    away_id = game.get("away_team_id", "")

    home_score = game.get("home_score")
    away_score = game.get("away_score")
    total_score = game.get("total_score")

    # If game already has oriented score keys, use them
    if "score" in game and "opp_score" in game:
        oriented = game.copy()
        if "total" not in oriented and total_score is not None:
            oriented["total"] = total_score
        return oriented

    is_home = (home_id == team_id)

    if is_home:
        score = home_score
        opp_score = away_score
    else:
        score = away_score
        opp_score = home_score

    winner = None
    if score is not None and opp_score is not None:
        winner = score > opp_score

    return {
        **game,
        "score": score,
        "opp_score": opp_score,
        "total": total_score,
        "home": is_home,
        "winner": winner,
        "possessions": game.get("possessions"),
    }

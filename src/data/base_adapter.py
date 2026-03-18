"""
Abstract base classes for all data adapters plus a factory that
instantiates the correct concrete implementation from config.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── Schedule adapter ──────────────────────────────────────────────────────────

class BaseScheduleAdapter(ABC):
    """
    Adapter interface for retrieving game schedules.
    All schedule adapters must implement this contract.
    """

    @abstractmethod
    def get_todays_schedule(self) -> list[dict]:
        """
        Return scheduled games for today.

        Returns:
            List of game dicts containing at minimum:
            game_id, date, home_team, home_team_id,
            away_team, away_team_id, neutral_site, status,
            home_score, away_score.
        """
        ...

    @abstractmethod
    def get_schedule_by_date(self, date: str) -> list[dict]:
        """
        Return scheduled games for a given date (YYYY-MM-DD).

        Args:
            date: Date string in YYYY-MM-DD format.

        Returns:
            Same structure as get_todays_schedule.
        """
        ...

    def get_completed_games_by_date_range(
        self, start: str, end: str
    ) -> list[dict]:
        """
        Return all completed games between start and end dates (inclusive).
        Default implementation iterates day-by-day.

        Args:
            start: Start date YYYY-MM-DD.
            end:   End date YYYY-MM-DD.

        Returns:
            Combined list of game dicts with status=='final'.
        """
        from datetime import date, timedelta

        results = []
        try:
            current = date.fromisoformat(start)
            end_d = date.fromisoformat(end)
        except ValueError:
            logger.error(f"Invalid date format: start={start} end={end}")
            return results

        while current <= end_d:
            date_str = current.strftime("%Y-%m-%d")
            try:
                games = self.get_schedule_by_date(date_str)
                completed = [g for g in games if g.get("status") == "final"]
                results.extend(completed)
            except Exception as exc:
                logger.warning(f"Could not fetch games for {date_str}: {exc}")
            current += timedelta(days=1)

        return results


# ── Stats adapter ─────────────────────────────────────────────────────────────

class BaseStatsAdapter(ABC):
    """
    Adapter interface for retrieving team statistical data.
    """

    @abstractmethod
    def get_team_stats(self, team_id: str, season: str) -> dict:
        """
        Return season aggregate stats for a team.

        Args:
            team_id: Canonical team identifier.
            season:  Season string e.g. '2024' or '2023-24'.

        Returns:
            Dict with offensive/defensive/tempo stats.
            Keys should match TeamSeasonStats columns.
        """
        ...

    @abstractmethod
    def get_game_log(self, team_id: str, season: str) -> list[dict]:
        """
        Return per-game results for a team in a given season.

        Args:
            team_id: Canonical team identifier.
            season:  Season string.

        Returns:
            List of game result dicts ordered by date ascending.
            Each dict contains: date, opponent_id, home, score,
            opp_score, total, possessions (if available).
        """
        ...

    def get_all_teams(self) -> list[dict]:
        """Return a list of all D1 teams with basic metadata."""
        return []


# ── Odds adapter ──────────────────────────────────────────────────────────────

class BaseOddsAdapter(ABC):
    """
    Adapter interface for retrieving betting odds.
    """

    @abstractmethod
    def get_current_odds(self, date: Optional[str] = None) -> list[dict]:
        """
        Return current totals odds for all available games.

        Args:
            date: Optional date filter (YYYY-MM-DD).

        Returns:
            List of odds dicts containing:
            game_id, home_team, away_team, commence_time,
            sportsbook, total, over_price, under_price.
        """
        ...

    @abstractmethod
    def get_odds_by_game(self, game_id: str) -> list[dict]:
        """
        Return all available sportsbook odds for a single game.

        Args:
            game_id: Unique game identifier.

        Returns:
            List of odds dicts (one per sportsbook).
        """
        ...

    def get_all_books_consensus(self, date: Optional[str] = None) -> list[dict]:
        """
        Return consensus (average) total across all available sportsbooks.

        Default implementation averages over get_current_odds results.
        """
        import statistics

        raw = self.get_current_odds(date)
        if not raw:
            return []

        # Group by game
        from collections import defaultdict
        groups: dict[str, list] = defaultdict(list)
        for entry in raw:
            groups[entry.get("game_id", "")].append(entry)

        consensus = []
        for game_id, book_odds in groups.items():
            totals = [o["total"] for o in book_odds if o.get("total")]
            if not totals:
                continue
            avg_total = statistics.mean(totals)
            rep = book_odds[0].copy()
            rep["sportsbook"] = "consensus"
            rep["total"] = round(avg_total * 2) / 2   # round to nearest 0.5
            consensus.append(rep)

        return consensus


# ── Adapter factory ───────────────────────────────────────────────────────────

class DataAdapterFactory:
    """
    Creates concrete adapter instances based on the application config.

    Usage::

        factory = DataAdapterFactory(config)
        schedule_adapter = factory.get_schedule_adapter()
        stats_adapter    = factory.get_stats_adapter()
        odds_adapter     = factory.get_odds_adapter()
    """

    def __init__(self, config: dict):
        self.config = config

    def get_schedule_adapter(self) -> BaseScheduleAdapter:
        source_cfg = self.config.get("data_sources", {}).get("schedule", {})
        primary = source_cfg.get("primary", "espn").lower()

        if primary == "espn":
            from src.data.espn_adapter import ESPNScheduleAdapter
            return ESPNScheduleAdapter(self.config)

        raise ValueError(f"Unknown schedule adapter: {primary!r}")

    def get_stats_adapter(self) -> BaseStatsAdapter:
        source_cfg = self.config.get("data_sources", {}).get("team_stats", {})
        primary = source_cfg.get("primary", "espn").lower()

        if primary == "espn":
            from src.data.espn_adapter import ESPNStatsAdapter
            return ESPNStatsAdapter(self.config)

        if primary == "torvik":
            from src.data.torvik_adapter import TorVikAdapter
            return TorVikAdapter(self.config)

        raise ValueError(f"Unknown stats adapter: {primary!r}")

    def get_odds_adapter(self) -> BaseOddsAdapter:
        source_cfg = self.config.get("data_sources", {}).get("odds", {})
        primary = source_cfg.get("primary", "the_odds_api").lower()

        if primary == "the_odds_api":
            from src.data.odds_adapter import TheOddsAPIAdapter
            return TheOddsAPIAdapter(self.config)

        raise ValueError(f"Unknown odds adapter: {primary!r}")

    def get_torvik_adapter(self):
        """Convenience accessor for the Torvik adapter regardless of config."""
        from src.data.torvik_adapter import TorVikAdapter
        return TorVikAdapter(self.config)

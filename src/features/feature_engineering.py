"""
Feature engineering for the CBB Totals Model.
Builds a flat feature dict for each game matchup.
"""

from __future__ import annotations

import math
from typing import Any, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── League averages (D1 NCAA 2023-24 approximate values) ──────────────────────
LEAGUE_AVG_OE = 105.0           # Points per 100 possessions
LEAGUE_AVG_DE = 105.0
LEAGUE_AVG_TEMPO = 68.5         # Possessions per 40 min
LEAGUE_AVG_PPP = 1.05           # Points per possession
HOME_COURT_ADVANTAGE = 3.5      # Points

# Sentinel value for missing features
MISSING_SENTINEL = -1.0


class FeatureEngineer:
    """
    Builds the complete feature vector for a single game matchup.

    Usage::

        fe = FeatureEngineer(config)
        features = fe.build_game_features(
            game, home_stats, away_stats, recent_home, recent_away
        )
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        model_cfg = self.config.get("model", {})
        self.league_avg_possessions: float = float(
            model_cfg.get("league_avg_possessions", LEAGUE_AVG_TEMPO)
        )
        self.league_avg_points_per_100: float = float(
            model_cfg.get("league_avg_points_per_100", LEAGUE_AVG_OE)
        )
        self.recent_form_windows: list[int] = model_cfg.get(
            "recent_form_windows", [3, 5, 10]
        )

    # ── Entry point ───────────────────────────────────────────────────────

    def build_game_features(
        self,
        game: dict,
        home_stats: dict,
        away_stats: dict,
        recent_home: list[dict],
        recent_away: list[dict],
    ) -> dict:
        """
        Build a complete flat feature dict for a single game.

        Args:
            game:        Game metadata dict (game_id, date, neutral_site, etc.)
            home_stats:  Season aggregate stats for home team.
            away_stats:  Season aggregate stats for away team.
            recent_home: List of recent game dicts for home team (newest first).
            recent_away: List of recent game dicts for away team (newest first).

        Returns:
            Flat dict of all feature values.  Missing values are filled with
            MISSING_SENTINEL (-1.0).  Includes 'data_completeness' key (0–1).
        """
        features: dict[str, Any] = {}

        # ── Tempo / pace features ─────────────────────────────────────────
        features.update(self._tempo_features(home_stats, away_stats, game))

        # ── Offensive features ────────────────────────────────────────────
        features.update(self._offensive_features(home_stats, "home"))
        features.update(self._offensive_features(away_stats, "away"))

        # ── Defensive features ────────────────────────────────────────────
        features.update(self._defensive_features(home_stats, "home"))
        features.update(self._defensive_features(away_stats, "away"))

        # ── Matchup interaction features ──────────────────────────────────
        features.update(self._matchup_features(features))

        # ── Recent form features ──────────────────────────────────────────
        for window in self.recent_form_windows:
            features.update(self._recent_form_features(recent_home, window, "home"))
            features.update(self._recent_form_features(recent_away, window, "away"))

        features.update(self._form_trend_features(recent_home, "home"))
        features.update(self._form_trend_features(recent_away, "away"))

        # ── Game environment features ─────────────────────────────────────
        features.update(self._environment_features(game, home_stats, away_stats))

        # ── Data completeness ─────────────────────────────────────────────
        features["data_completeness"] = self._compute_completeness(features)

        return features

    # ── Tempo / pace ──────────────────────────────────────────────────────

    def _tempo_features(self, home_stats: dict, away_stats: dict, game: dict) -> dict:
        home_tempo = _get(home_stats, "adj_tempo", LEAGUE_AVG_TEMPO)
        away_tempo = _get(away_stats, "adj_tempo", LEAGUE_AVG_TEMPO)

        # Possessions formula: each team runs half the possessions
        # When a fast team meets a slow team, pace regresses toward the mean
        # Formula from KenPom: possessions = (home_tempo + away_tempo) / 2
        # adjusted by league average
        raw_expected = (home_tempo + away_tempo) / 2.0
        league_factor = self.league_avg_possessions / LEAGUE_AVG_TEMPO
        expected_possessions = raw_expected * league_factor

        tempo_diff = home_tempo - away_tempo
        # Pace tendency: >70 = "fast", <66 = "slow"
        pace_tendency_home = 1.0 if home_tempo >= 70.0 else (-1.0 if home_tempo <= 66.0 else 0.0)
        pace_tendency_away = 1.0 if away_tempo >= 70.0 else (-1.0 if away_tempo <= 66.0 else 0.0)

        return {
            "home_adj_tempo": home_tempo,
            "away_adj_tempo": away_tempo,
            "expected_possessions": expected_possessions,
            "tempo_differential": abs(tempo_diff),
            "tempo_diff_signed": tempo_diff,
            "pace_tendency_home": pace_tendency_home,
            "pace_tendency_away": pace_tendency_away,
        }

    # ── Offensive features ────────────────────────────────────────────────

    def _offensive_features(self, stats: dict, side: str) -> dict:
        return {
            f"adj_oe_{side}": _get(stats, "adj_oe", LEAGUE_AVG_OE),
            f"raw_oe_{side}": _get(stats, "raw_oe", _get(stats, "ppg", MISSING_SENTINEL)),
            f"efg_pct_{side}": _get(stats, "efg_pct", MISSING_SENTINEL),
            f"two_p_pct_{side}": _get(stats, "two_p_pct", MISSING_SENTINEL),
            f"three_p_pct_{side}": _get(stats, "three_p_pct", MISSING_SENTINEL),
            f"three_pa_rate_{side}": _get(stats, "three_pa_rate", MISSING_SENTINEL),
            f"ft_rate_{side}": _get(stats, "ft_rate", MISSING_SENTINEL),
            f"tov_rate_{side}": _get(stats, "tov_rate", MISSING_SENTINEL),
            f"orb_rate_{side}": _get(stats, "orb_rate", MISSING_SENTINEL),
            f"ppg_{side}": _get(stats, "ppg", MISSING_SENTINEL),
        }

    # ── Defensive features ────────────────────────────────────────────────

    def _defensive_features(self, stats: dict, side: str) -> dict:
        return {
            f"adj_de_{side}": _get(stats, "adj_de", LEAGUE_AVG_DE),
            f"opp_efg_pct_{side}": _get(stats, "opp_efg_pct", MISSING_SENTINEL),
            f"opp_two_p_pct_{side}": _get(stats, "opp_two_p_pct", MISSING_SENTINEL),
            f"opp_three_p_pct_{side}": _get(stats, "opp_three_p_pct", MISSING_SENTINEL),
            f"opp_three_pa_rate_{side}": _get(stats, "opp_three_pa_rate", MISSING_SENTINEL),
            f"forced_tov_rate_{side}": _get(stats, "opp_tov_rate", MISSING_SENTINEL),
            f"drb_rate_{side}": _get(stats, "drb_rate", MISSING_SENTINEL),
            f"opp_ft_rate_{side}": _get(stats, "opp_ft_rate", MISSING_SENTINEL),
            f"opp_ppg_{side}": _get(stats, "opp_ppg", MISSING_SENTINEL),
        }

    # ── Matchup interaction features ──────────────────────────────────────

    def _matchup_features(self, f: dict) -> dict:
        """
        Compute features that represent the interaction between both teams.
        f is the partial features dict built so far.
        """
        results = {}

        adj_oe_home = f.get("adj_oe_home", LEAGUE_AVG_OE)
        adj_oe_away = f.get("adj_oe_away", LEAGUE_AVG_OE)
        adj_de_home = f.get("adj_de_home", LEAGUE_AVG_DE)
        adj_de_away = f.get("adj_de_away", LEAGUE_AVG_DE)

        league_sq = (LEAGUE_AVG_OE * LEAGUE_AVG_DE)

        # How good is home offense vs away defense (normalized)
        results["off_vs_def_home"] = (adj_oe_home * adj_de_away) / league_sq
        results["off_vs_def_away"] = (adj_oe_away * adj_de_home) / league_sq

        # Combined offensive firepower vs combined defense
        results["combined_oe"] = (adj_oe_home + adj_oe_away) / 2.0
        results["combined_de"] = (adj_de_home + adj_de_away) / 2.0
        results["oe_de_ratio"] = results["combined_oe"] / max(results["combined_de"], 1.0)

        # 3PA matchup (high rate = more variance)
        three_pa_home = f.get("three_pa_rate_home", MISSING_SENTINEL)
        three_pa_away = f.get("three_pa_rate_away", MISSING_SENTINEL)
        results["three_pa_matchup"] = _safe_avg(
            [v for v in [three_pa_home, three_pa_away] if v != MISSING_SENTINEL]
        )

        # Rebounding battle: home offensive rebounds vs away defensive
        orb_home = f.get("orb_rate_home", MISSING_SENTINEL)
        drb_away = f.get("drb_rate_away", MISSING_SENTINEL)
        orb_away = f.get("orb_rate_away", MISSING_SENTINEL)
        drb_home = f.get("drb_rate_home", MISSING_SENTINEL)
        results["rebounding_battle_home"] = _diff_or_missing(orb_home, drb_away)
        results["rebounding_battle_away"] = _diff_or_missing(orb_away, drb_home)

        # Turnover environment
        tov_home = f.get("tov_rate_home", MISSING_SENTINEL)
        tov_away = f.get("tov_rate_away", MISSING_SENTINEL)
        results["tov_environment"] = _safe_avg(
            [v for v in [tov_home, tov_away] if v != MISSING_SENTINEL]
        )
        results["tov_differential"] = _diff_or_missing(tov_home, tov_away)

        # Free throw environment
        ft_home = f.get("ft_rate_home", MISSING_SENTINEL)
        ft_away = f.get("ft_rate_away", MISSING_SENTINEL)
        results["ft_environment"] = _safe_avg(
            [v for v in [ft_home, ft_away] if v != MISSING_SENTINEL]
        )

        # Efficiency matchup summary
        results["efficiency_sum"] = (adj_oe_home + adj_oe_away) / 2.0

        return results

    # ── Recent form features ──────────────────────────────────────────────

    def _recent_form_features(
        self, games: list[dict], window: int, side: str
    ) -> dict:
        """
        Compute rolling stats over the last N games.

        Args:
            games:  List of game result dicts (newest first).
            window: Rolling window size.
            side:   'home' or 'away'.

        Returns:
            Dict of rolling stat features.
        """
        prefix = f"{side}_last_{window}"
        results = {}

        if not games:
            return {
                f"{prefix}_pts": MISSING_SENTINEL,
                f"{prefix}_opp_pts": MISSING_SENTINEL,
                f"{prefix}_total": MISSING_SENTINEL,
                f"{prefix}_pace": MISSING_SENTINEL,
                f"{prefix}_off_eff": MISSING_SENTINEL,
                f"{prefix}_def_eff": MISSING_SENTINEL,
            }

        recent = games[:window]

        pts = [g.get("score") for g in recent if g.get("score") is not None]
        opp_pts = [g.get("opp_score") for g in recent if g.get("opp_score") is not None]
        totals = [g.get("total") for g in recent if g.get("total") is not None]
        poss_list = [g.get("possessions") for g in recent if g.get("possessions") is not None]

        results[f"{prefix}_pts"] = _safe_avg(pts)
        results[f"{prefix}_opp_pts"] = _safe_avg(opp_pts)
        results[f"{prefix}_total"] = _safe_avg(totals)

        # Pace: if possessions available directly, use them; else estimate from total
        if poss_list:
            results[f"{prefix}_pace"] = _safe_avg(poss_list)
        elif totals:
            # Estimate possessions from scoring: pts ~ poss * PPP
            # poss ≈ pts / league_ppp (rough)
            estimated_poss = [t / (2 * LEAGUE_AVG_PPP) for t in totals]
            results[f"{prefix}_pace"] = _safe_avg(estimated_poss)
        else:
            results[f"{prefix}_pace"] = MISSING_SENTINEL

        # Offensive efficiency = pts / estimated_poss
        if pts and results.get(f"{prefix}_pace", MISSING_SENTINEL) != MISSING_SENTINEL:
            pace_val = results[f"{prefix}_pace"]
            if pace_val and pace_val > 0:
                avg_pts = _safe_avg(pts)
                results[f"{prefix}_off_eff"] = (avg_pts / pace_val) * 100.0
            else:
                results[f"{prefix}_off_eff"] = MISSING_SENTINEL
        else:
            results[f"{prefix}_off_eff"] = MISSING_SENTINEL

        # Defensive efficiency = opp_pts / estimated_poss
        if opp_pts and results.get(f"{prefix}_pace", MISSING_SENTINEL) != MISSING_SENTINEL:
            pace_val = results[f"{prefix}_pace"]
            if pace_val and pace_val > 0:
                avg_opp_pts = _safe_avg(opp_pts)
                results[f"{prefix}_def_eff"] = (avg_opp_pts / pace_val) * 100.0
            else:
                results[f"{prefix}_def_eff"] = MISSING_SENTINEL
        else:
            results[f"{prefix}_def_eff"] = MISSING_SENTINEL

        return results

    # ── Form trend ────────────────────────────────────────────────────────

    def _form_trend_features(self, games: list[dict], side: str) -> dict:
        """
        Compute form trend: last-3 offensive efficiency vs last-10.

        Positive = team trending up; negative = trending down.
        """
        if not games or len(games) < 3:
            return {
                f"{side}_form_trend": MISSING_SENTINEL,
                f"{side}_scoring_trend": MISSING_SENTINEL,
            }

        recent_3 = games[:3]
        recent_10 = games[:10] if len(games) >= 10 else games

        pts_3 = [g.get("score") for g in recent_3 if g.get("score") is not None]
        pts_10 = [g.get("score") for g in recent_10 if g.get("score") is not None]

        avg_3 = _safe_avg(pts_3)
        avg_10 = _safe_avg(pts_10)

        if avg_3 == MISSING_SENTINEL or avg_10 == MISSING_SENTINEL or avg_10 == 0:
            return {
                f"{side}_form_trend": MISSING_SENTINEL,
                f"{side}_scoring_trend": MISSING_SENTINEL,
            }

        trend = avg_3 - avg_10
        scoring_trend = (avg_3 - avg_10) / avg_10  # Percentage change

        return {
            f"{side}_form_trend": trend,
            f"{side}_scoring_trend": scoring_trend,
        }

    # ── Environment features ──────────────────────────────────────────────

    def _environment_features(
        self, game: dict, home_stats: dict, away_stats: dict
    ) -> dict:
        neutral = bool(game.get("neutral_site", False))
        home_court_adj = 0.0 if neutral else HOME_COURT_ADVANTAGE

        # Days rest — from game dict if precomputed, else use 0
        days_rest_home = int(game.get("days_rest_home", 0) or 0)
        days_rest_away = int(game.get("days_rest_away", 0) or 0)
        rest_differential = days_rest_home - days_rest_away

        # Conference game detection
        conf_home = str(home_stats.get("conference", "") or "")
        conf_away = str(away_stats.get("conference", "") or "")
        is_conference_game = int(conf_home == conf_away and bool(conf_home))

        # Strength of schedule
        sos_home = _get(home_stats, "sos", MISSING_SENTINEL)
        sos_away = _get(away_stats, "sos", MISSING_SENTINEL)
        sos_diff = _diff_or_missing(sos_home, sos_away)

        # Encode conference as a simple hash (for tree-based models)
        conf_home_enc = _encode_conference(conf_home)
        conf_away_enc = _encode_conference(conf_away)

        return {
            "neutral_site": int(neutral),
            "home_court_advantage_adj": home_court_adj,
            "days_rest_home": days_rest_home,
            "days_rest_away": days_rest_away,
            "rest_differential": rest_differential,
            "is_conference_game": is_conference_game,
            "sos_home": sos_home,
            "sos_away": sos_away,
            "sos_differential": sos_diff,
            "conference_home_enc": conf_home_enc,
            "conference_away_enc": conf_away_enc,
        }

    # ── Data completeness ─────────────────────────────────────────────────

    def _compute_completeness(self, features: dict) -> float:
        """
        Compute what fraction of features have real (non-sentinel) values.

        Returns a float in [0, 1].
        """
        # Exclude meta-features from the completeness calculation
        exclude = {"data_completeness", "neutral_site", "is_conference_game",
                   "home_court_advantage_adj", "days_rest_home", "days_rest_away",
                   "rest_differential", "conference_home_enc", "conference_away_enc"}

        scorable = {k: v for k, v in features.items() if k not in exclude}
        if not scorable:
            return 0.0

        real_count = sum(
            1 for v in scorable.values()
            if v is not None and v != MISSING_SENTINEL and not _is_nan(v)
        )
        return real_count / len(scorable)


# ── Shared feature column list (used by ML models) ────────────────────────────

FEATURE_COLUMNS: list[str] = [
    # Tempo
    "home_adj_tempo", "away_adj_tempo", "expected_possessions",
    "tempo_differential", "tempo_diff_signed",
    "pace_tendency_home", "pace_tendency_away",
    # Offense
    "adj_oe_home", "adj_oe_away",
    "raw_oe_home", "raw_oe_away",
    "efg_pct_home", "efg_pct_away",
    "two_p_pct_home", "two_p_pct_away",
    "three_p_pct_home", "three_p_pct_away",
    "three_pa_rate_home", "three_pa_rate_away",
    "ft_rate_home", "ft_rate_away",
    "tov_rate_home", "tov_rate_away",
    "orb_rate_home", "orb_rate_away",
    "ppg_home", "ppg_away",
    # Defense
    "adj_de_home", "adj_de_away",
    "opp_efg_pct_home", "opp_efg_pct_away",
    "opp_two_p_pct_home", "opp_two_p_pct_away",
    "opp_three_p_pct_home", "opp_three_p_pct_away",
    "opp_three_pa_rate_home", "opp_three_pa_rate_away",
    "forced_tov_rate_home", "forced_tov_rate_away",
    "drb_rate_home", "drb_rate_away",
    "opp_ft_rate_home", "opp_ft_rate_away",
    "opp_ppg_home", "opp_ppg_away",
    # Matchup
    "off_vs_def_home", "off_vs_def_away",
    "combined_oe", "combined_de", "oe_de_ratio",
    "three_pa_matchup",
    "rebounding_battle_home", "rebounding_battle_away",
    "tov_environment", "tov_differential",
    "ft_environment", "efficiency_sum",
    # Recent form (3, 5, 10 window)
    "home_last_3_pts", "home_last_3_opp_pts", "home_last_3_total",
    "home_last_3_pace", "home_last_3_off_eff", "home_last_3_def_eff",
    "away_last_3_pts", "away_last_3_opp_pts", "away_last_3_total",
    "away_last_3_pace", "away_last_3_off_eff", "away_last_3_def_eff",
    "home_last_5_pts", "home_last_5_opp_pts", "home_last_5_total",
    "home_last_5_pace", "home_last_5_off_eff", "home_last_5_def_eff",
    "away_last_5_pts", "away_last_5_opp_pts", "away_last_5_total",
    "away_last_5_pace", "away_last_5_off_eff", "away_last_5_def_eff",
    "home_last_10_pts", "home_last_10_opp_pts", "home_last_10_total",
    "home_last_10_pace", "home_last_10_off_eff", "home_last_10_def_eff",
    "away_last_10_pts", "away_last_10_opp_pts", "away_last_10_total",
    "away_last_10_pace", "away_last_10_off_eff", "away_last_10_def_eff",
    # Form trend
    "home_form_trend", "home_scoring_trend",
    "away_form_trend", "away_scoring_trend",
    # Environment
    "neutral_site", "home_court_advantage_adj",
    "days_rest_home", "days_rest_away", "rest_differential",
    "is_conference_game",
    "sos_home", "sos_away", "sos_differential",
    "conference_home_enc", "conference_away_enc",
    # Meta
    "data_completeness",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get(d: dict, key: str, default: float = MISSING_SENTINEL) -> float:
    """Safe dict get with numeric fallback."""
    v = d.get(key)
    if v is None or _is_nan(v):
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _safe_avg(values: list) -> float:
    """Mean of a list; returns MISSING_SENTINEL if list is empty."""
    if not values:
        return MISSING_SENTINEL
    valid = [float(v) for v in values if v is not None and not _is_nan(v)]
    if not valid:
        return MISSING_SENTINEL
    return sum(valid) / len(valid)


def _diff_or_missing(a: float, b: float) -> float:
    """Compute a - b if both are real values, else MISSING_SENTINEL."""
    if a == MISSING_SENTINEL or b == MISSING_SENTINEL:
        return MISSING_SENTINEL
    if _is_nan(a) or _is_nan(b):
        return MISSING_SENTINEL
    return a - b


def _is_nan(v) -> bool:
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return False


_CONFERENCE_MAP: dict[str, int] = {
    "acc": 1, "big ten": 2, "big 12": 3, "sec": 4, "pac-12": 5,
    "big east": 6, "aac": 7, "atlantic 10": 8, "mwc": 9, "wcc": 10,
    "cusa": 11, "sun belt": 12, "mac": 13, "colonial": 14, "horizon": 15,
    "ivy": 16, "maac": 17, "mvc": 18, "ohio valley": 19, "patriot": 20,
    "southern": 21, "southland": 22, "swac": 23, "meac": 24, "big south": 25,
    "big sky": 26, "big west": 27, "summit": 28, "america east": 29,
}


def _encode_conference(conf_name: str) -> int:
    """Encode conference name as integer (0 = unknown)."""
    if not conf_name:
        return 0
    return _CONFERENCE_MAP.get(conf_name.lower().strip(), 0)

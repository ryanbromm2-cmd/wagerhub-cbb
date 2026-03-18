"""
Physics-based baseline model for CBB game totals.
Uses possessions × efficiency to project scores.
"""

from __future__ import annotations

import math
from typing import Any, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── League constants (NCAA D1 2023-24) ────────────────────────────────────────
LEAGUE_AVG_OE = 105.0          # Points per 100 possessions (offense)
LEAGUE_AVG_DE = 105.0          # Points per 100 possessions (defense)
LEAGUE_AVG_TEMPO = 68.5        # Possessions per 40 min
LEAGUE_AVG_PPP = LEAGUE_AVG_OE / 100.0   # ~1.05 points per possession

# Home court advantage splits
HOME_COURT_TOTAL = 3.5         # Total home advantage (in points)
HOME_COURT_HOME = HOME_COURT_TOTAL * 0.5    # Added to home score
HOME_COURT_AWAY = HOME_COURT_TOTAL * 0.5    # Subtracted from away score

# Pace regression factor: when tempos differ, regress toward mean
PACE_REGRESSION_FACTOR = 0.5   # 0=no regression, 1=full regression to mean

# Thresholds for conservative blending
MAX_RAW_TOTAL = 200.0
MIN_RAW_TOTAL = 90.0

MISSING = -1.0


class BaselineModel:
    """
    Physics-based game total projector.

    Core formula::

        projected_possessions = blend(home_tempo, away_tempo)
        home_ppp = (home_adj_oe / L_oe) * (away_adj_de / L_de) * L_ppp
        away_ppp = (away_adj_oe / L_oe) * (home_adj_de / L_de) * L_ppp
        home_score = home_ppp * possessions  (± home court adj)
        away_score = away_ppp * possessions
        total      = home_score + away_score

    All adjustments (FT, TOV, 3PA) are multiplicative corrections applied
    to the base efficiency estimate.
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        model_cfg = self.config.get("model", {})

        self.league_avg_oe: float = float(LEAGUE_AVG_OE)
        self.league_avg_de: float = float(LEAGUE_AVG_DE)
        self.league_avg_tempo: float = float(
            model_cfg.get("league_avg_possessions", LEAGUE_AVG_TEMPO)
        )
        self.league_ppp: float = self.league_avg_oe / 100.0

        self.baseline_weight: float = float(
            model_cfg.get("baseline_weight", 0.35)
        )
        self.home_court_advantage: float = HOME_COURT_TOTAL

    # ── Entry point ───────────────────────────────────────────────────────

    def predict(self, features: dict) -> dict:
        """
        Generate a baseline projection for a single game.

        Args:
            features: Feature dict from FeatureEngineer.build_game_features().

        Returns:
            Dict with keys:
            - baseline_total (float)
            - predicted_home (float)
            - predicted_away (float)
            - predicted_possessions (float)
            - baseline_confidence (float 0–1)
            - home_ppp (float)
            - away_ppp (float)
        """
        # ── Extract key features ──────────────────────────────────────────
        home_adj_oe = _feat(features, "adj_oe_home", LEAGUE_AVG_OE)
        away_adj_oe = _feat(features, "adj_oe_away", LEAGUE_AVG_OE)
        home_adj_de = _feat(features, "adj_de_home", LEAGUE_AVG_DE)
        away_adj_de = _feat(features, "adj_de_away", LEAGUE_AVG_DE)
        home_tempo = _feat(features, "home_adj_tempo", self.league_avg_tempo)
        away_tempo = _feat(features, "away_adj_tempo", self.league_avg_tempo)

        neutral_site = bool(int(_feat(features, "neutral_site", 0)))
        tov_home = _feat(features, "tov_rate_home", None)
        tov_away = _feat(features, "tov_rate_away", None)
        ft_home = _feat(features, "ft_rate_home", None)
        ft_away = _feat(features, "ft_rate_away", None)
        data_completeness = float(_feat(features, "data_completeness", 0.5))

        # ── Step 1: Projected possessions ────────────────────────────────
        projected_possessions = self._project_possessions(home_tempo, away_tempo)

        # ── Step 2: Points per possession for each team ───────────────────
        # Home team offense vs away team defense
        home_ppp = self._compute_ppp(
            home_adj_oe, away_adj_de, tov_home, ft_away
        )
        # Away team offense vs home team defense
        away_ppp = self._compute_ppp(
            away_adj_oe, home_adj_de, tov_away, ft_home
        )

        # ── Step 3: Raw scores ────────────────────────────────────────────
        raw_home_score = home_ppp * projected_possessions
        raw_away_score = away_ppp * projected_possessions

        # ── Step 4: Home court adjustment ────────────────────────────────
        if neutral_site:
            home_adj = 0.0
            away_adj = 0.0
        else:
            home_adj = HOME_COURT_HOME
            away_adj = -HOME_COURT_AWAY

        home_score = raw_home_score + home_adj
        away_score = raw_away_score + away_adj

        # ── Step 5: Sanity clamp ──────────────────────────────────────────
        home_score = _clamp(home_score, 40.0, 130.0)
        away_score = _clamp(away_score, 40.0, 130.0)
        total = home_score + away_score

        # ── Step 6: Confidence ────────────────────────────────────────────
        baseline_confidence = self._compute_confidence(
            data_completeness, home_adj_oe, away_adj_oe, home_adj_de, away_adj_de
        )

        result = {
            "baseline_total": round(total, 2),
            "predicted_home": round(home_score, 2),
            "predicted_away": round(away_score, 2),
            "predicted_possessions": round(projected_possessions, 2),
            "baseline_confidence": round(baseline_confidence, 3),
            "home_ppp": round(home_ppp, 4),
            "away_ppp": round(away_ppp, 4),
            "home_adj_applied": home_adj,
            "neutral_site": neutral_site,
        }

        logger.debug(
            f"Baseline: possessions={projected_possessions:.1f}, "
            f"home={home_score:.1f}, away={away_score:.1f}, "
            f"total={total:.1f}, conf={baseline_confidence:.2f}"
        )
        return result

    # ── Possessions formula ───────────────────────────────────────────────

    def _project_possessions(self, home_tempo: float, away_tempo: float) -> float:
        """
        Project expected possessions per team.

        Uses a blended pace: average of both tempos, adjusted by league mean.
        When teams have very different tempos (diff > 5), apply regression
        toward the mean to reflect that pace is partly a function of the
        slower team's defensive pace control.

        Formula::

            raw_avg = (home_tempo + away_tempo) / 2
            divergence = abs(home_tempo - away_tempo)
            if divergence > 5:
                regression = (divergence - 5) / 20 * REGRESSION_FACTOR
                raw_avg = raw_avg * (1 - regression) + league_avg * regression
            pace_factor = league_avg / league_avg  = 1.0 (normalized)
            possessions = raw_avg
        """
        raw_avg = (home_tempo + away_tempo) / 2.0
        divergence = abs(home_tempo - away_tempo)

        if divergence > 5.0:
            # Regress toward league average
            regression_amount = min((divergence - 5.0) / 20.0, 0.5) * PACE_REGRESSION_FACTOR
            raw_avg = raw_avg * (1.0 - regression_amount) + self.league_avg_tempo * regression_amount

        return raw_avg

    # ── Points per possession ─────────────────────────────────────────────

    def _compute_ppp(
        self,
        offense_adj_oe: float,
        defense_adj_de: float,
        tov_rate: Optional[float] = None,
        opp_ft_rate: Optional[float] = None,
    ) -> float:
        """
        Compute adjusted points per possession.

        Base formula (KenPom-style multiplicative):
            ppp = (adj_oe / L_oe) * (adj_de / L_de) * L_ppp

        Adjustments:
        - Turnover: high TOV rate reduces effective possessions → lower scoring
        - Free throws: high FT rate inflates scoring slightly

        Args:
            offense_adj_oe: Offensive team's adjusted offensive efficiency.
            defense_adj_de: Defensive team's adjusted defensive efficiency.
            tov_rate:       Offensive team's turnover rate (turnovers/possession).
            opp_ft_rate:    Opponent's free throw rate (FTA/FGA).

        Returns:
            Estimated points per possession.
        """
        # Normalize against league averages
        oe_factor = offense_adj_oe / self.league_avg_oe
        de_factor = defense_adj_de / self.league_avg_de

        base_ppp = oe_factor * de_factor * self.league_ppp

        # Turnover adjustment: each extra 1% TOV rate above league average
        # reduces scoring by approximately 0.002 pts/possession
        # League avg TOV rate ≈ 18% (0.18)
        if tov_rate is not None and tov_rate != MISSING and tov_rate > 0:
            league_avg_tov = 0.18
            tov_delta = tov_rate - league_avg_tov
            tov_adj = 1.0 - (tov_delta * 0.3)  # 0.3 = sensitivity
            tov_adj = _clamp(tov_adj, 0.85, 1.10)
            base_ppp *= tov_adj

        # Free throw rate adjustment: high FTR → more free throw scoring
        # League avg FTR ≈ 0.30 (FTA/FGA)
        if opp_ft_rate is not None and opp_ft_rate != MISSING and opp_ft_rate > 0:
            league_avg_ftr = 0.30
            ftr_delta = opp_ft_rate - league_avg_ftr
            ftr_adj = 1.0 + (ftr_delta * 0.05)
            ftr_adj = _clamp(ftr_adj, 0.95, 1.05)
            base_ppp *= ftr_adj

        return _clamp(base_ppp, 0.7, 1.5)

    # ── Confidence ────────────────────────────────────────────────────────

    def _compute_confidence(
        self,
        data_completeness: float,
        home_adj_oe: float,
        away_adj_oe: float,
        home_adj_de: float,
        away_adj_de: float,
    ) -> float:
        """
        Estimate baseline model confidence.

        Factors:
        1. Data completeness (from feature dict)
        2. Whether key efficiency metrics are non-default (i.e., real data)

        Returns float in [0, 1].
        """
        conf = data_completeness

        # Penalize if efficiency figures are exactly at league average
        # (may indicate placeholder/default values)
        all_at_default = all(
            abs(v - LEAGUE_AVG_OE) < 0.1
            for v in [home_adj_oe, away_adj_oe, home_adj_de, away_adj_de]
        )
        if all_at_default:
            conf *= 0.5

        # Boost if we have good data
        if data_completeness > 0.85:
            conf = min(conf * 1.1, 1.0)

        return _clamp(conf, 0.0, 1.0)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _feat(features: dict, key: str, default) -> Any:
    """Safe feature getter that treats MISSING sentinel as None."""
    v = features.get(key, default)
    if v is None:
        return default
    try:
        f = float(v)
        if math.isnan(f) or f == MISSING:
            return default
        return f
    except (TypeError, ValueError):
        return default


def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp value to [lo, hi]."""
    return max(lo, min(hi, value))

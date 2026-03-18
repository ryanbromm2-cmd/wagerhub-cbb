"""
Ensemble model that blends baseline and ML predictions into a final projection.
Also computes a composite confidence score.
"""

from __future__ import annotations

import math
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Threshold at which baseline and ML diverge enough to regress toward mean
_LARGE_DISAGREEMENT_THRESHOLD = 8.0

# NCAA D1 approximate season-average total (used for regression toward mean)
_LEAGUE_AVG_TOTAL = 145.0

MISSING = -1.0


class EnsembleModel:
    """
    Blends the baseline physics model with the ML ensemble prediction.

    Configuration keys used (from config['model']):
    - baseline_weight       (default 0.35)
    - ml_weight             (default 0.65)
    - ensemble.xgboost_weight, .lightgbm_weight, .ridge_weight

    Configuration keys used (from config['confidence']):
    - data_completeness_weight  (default 0.25)
    - model_agreement_weight    (default 0.35)
    - edge_magnitude_weight     (default 0.25)
    - line_movement_weight      (default 0.15)
    - conservative_discount     (default 0.85)
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        model_cfg = self.config.get("model", {})
        conf_cfg = self.config.get("confidence", {})

        self.baseline_weight: float = float(model_cfg.get("baseline_weight", 0.35))
        self.ml_weight: float = float(model_cfg.get("ml_weight", 0.65))

        # Confidence factor weights
        self.dc_weight: float = float(conf_cfg.get("data_completeness_weight", 0.25))
        self.ma_weight: float = float(conf_cfg.get("model_agreement_weight", 0.35))
        self.em_weight: float = float(conf_cfg.get("edge_magnitude_weight", 0.25))
        self.lm_weight: float = float(conf_cfg.get("line_movement_weight", 0.15))
        self.conservative_discount: float = float(
            conf_cfg.get("conservative_discount", 0.85)
        )

    # ── Core predict ──────────────────────────────────────────────────────

    def predict(
        self,
        baseline_result: dict,
        ml_result: dict,
        config: dict = None,
    ) -> dict:
        """
        Blend baseline and ML results into a final projection.

        Args:
            baseline_result: Output dict from BaselineModel.predict().
            ml_result:       Output dict from MLModelPredictor.predict().
                             Can be empty dict if ML is unavailable.
            config:          Optional config override (uses self.config if None).

        Returns:
            Dict with:
            - ensemble_total (float)
            - predicted_home_score (float)
            - predicted_away_score (float)
            - predicted_possessions (float)
            - baseline_contribution (float) — weighted baseline component
            - ml_contribution (float) — weighted ML component
            - model_versions_used (list[str])
            - blend_mode (str) — 'full' | 'baseline_only' | 'conservative'
        """
        cfg = config or self.config

        baseline_total = _get_float(baseline_result, "baseline_total")
        ml_total = _get_float(ml_result, "ml_ensemble_total")
        baseline_home = _get_float(baseline_result, "predicted_home")
        baseline_away = _get_float(baseline_result, "predicted_away")
        predicted_possessions = _get_float(baseline_result, "predicted_possessions", 68.5)

        model_cfg = cfg.get("model", {})
        bw = float(model_cfg.get("baseline_weight", self.baseline_weight))
        mlw = float(model_cfg.get("ml_weight", self.ml_weight))

        versions_used = ["baseline"]
        blend_mode = "baseline_only"

        if ml_total is not None and ml_total > 0:
            # Check for large disagreement
            disagreement = abs(baseline_total - ml_total) if baseline_total else 0.0

            if disagreement > _LARGE_DISAGREEMENT_THRESHOLD:
                # Conservative: regress both toward league mean before blending
                logger.debug(
                    f"Baseline ({baseline_total:.1f}) and ML ({ml_total:.1f}) "
                    f"disagree by {disagreement:.1f} pts — applying conservative blend."
                )
                baseline_adj = (
                    baseline_total * 0.7 + _LEAGUE_AVG_TOTAL * 0.3
                )
                ml_adj = ml_total * 0.7 + _LEAGUE_AVG_TOTAL * 0.3
                ensemble_total = bw * baseline_adj + mlw * ml_adj
                blend_mode = "conservative"
            else:
                ensemble_total = bw * baseline_total + mlw * ml_total
                blend_mode = "full"

            versions_used.append("ml_ensemble")
        else:
            # ML not available — use baseline only
            ensemble_total = baseline_total
            blend_mode = "baseline_only"

        # Clamp total to reasonable range
        ensemble_total = _clamp(ensemble_total, 100.0, 210.0)

        # Derive home/away splits from ensemble total
        # If we have ML we adjust the baseline split proportionally
        if blend_mode in ("full", "conservative") and baseline_home and baseline_away:
            baseline_sum = baseline_home + baseline_away
            if baseline_sum > 0:
                home_frac = baseline_home / baseline_sum
                away_frac = baseline_away / baseline_sum
            else:
                home_frac = 0.5
                away_frac = 0.5
            predicted_home = ensemble_total * home_frac
            predicted_away = ensemble_total * away_frac
        else:
            predicted_home = baseline_home or ensemble_total / 2.0
            predicted_away = baseline_away or ensemble_total / 2.0

        return {
            "ensemble_total": round(ensemble_total, 2),
            "predicted_home_score": round(predicted_home, 2),
            "predicted_away_score": round(predicted_away, 2),
            "predicted_possessions": round(predicted_possessions, 2),
            "baseline_total": round(baseline_total, 2) if baseline_total else None,
            "ml_total": round(ml_total, 2) if ml_total else None,
            "baseline_contribution": round(bw * baseline_total, 2) if baseline_total else None,
            "ml_contribution": round(mlw * ml_total, 2) if ml_total else None,
            "model_versions_used": versions_used,
            "blend_mode": blend_mode,
        }

    # ── Confidence score ──────────────────────────────────────────────────

    def compute_confidence_score(
        self,
        features: dict,
        baseline_result: dict,
        ml_result: dict,
        odds_context: Optional[dict] = None,
    ) -> float:
        """
        Compute a composite confidence score (0–1) for the projection.

        Factors:
        1. Data completeness: fraction of features with real values.
        2. Model agreement: how closely baseline and ML models agree.
        3. Edge magnitude: placeholder (0.5 default — computed externally).
        4. Line movement: if odds moved toward our projection, boost confidence.

        Args:
            features:         Feature dict (for data_completeness).
            baseline_result:  BaselineModel.predict() output.
            ml_result:        MLModelPredictor.predict() output (or empty dict).
            odds_context:     Optional dict with 'opening_total', 'current_total'.

        Returns:
            Float in [0, 1].
        """
        # ── Factor 1: Data completeness ───────────────────────────────────
        data_completeness = float(
            features.get("data_completeness", 0.5) or 0.5
        )
        dc_score = _clamp(data_completeness, 0.0, 1.0)

        # ── Factor 2: Model agreement ──────────────────────────────────────
        baseline_total = _get_float(baseline_result, "baseline_total")
        ml_total = _get_float(ml_result, "ml_ensemble_total")
        ml_model_agreement = _get_float(ml_result, "model_agreement_score", 0.5)

        if baseline_total and ml_total:
            diff = abs(baseline_total - ml_total)
            # Scale: 0 diff = 1.0 agreement, 10+ diff = 0.0
            cross_model_agreement = max(0.0, 1.0 - diff / 10.0)
            agreement_score = (cross_model_agreement + float(ml_model_agreement)) / 2.0
        elif ml_model_agreement is not None:
            agreement_score = float(ml_model_agreement)
        else:
            agreement_score = 0.5

        # ── Factor 3: Edge magnitude (placeholder — set externally) ───────
        # This factor will be updated when the edge is computed in EdgeCalculator.
        # Default 0.5 (neutral).
        edge_magnitude_score = 0.5

        # ── Factor 4: Line movement ────────────────────────────────────────
        lm_score = 0.5  # Neutral default
        if odds_context:
            opening = odds_context.get("opening_total")
            current = odds_context.get("current_total")
            model_total = (
                (baseline_total + ml_total) / 2.0
                if baseline_total and ml_total
                else (baseline_total or ml_total or 145.0)
            )
            if opening is not None and current is not None and model_total:
                opening_diff = opening - model_total
                current_diff = current - model_total
                # If the line moved toward our projection, boost
                if abs(current_diff) < abs(opening_diff):
                    lm_score = 0.7
                elif abs(current_diff) > abs(opening_diff):
                    lm_score = 0.3
                else:
                    lm_score = 0.5

        # ── Weighted composite ─────────────────────────────────────────────
        total_weight = self.dc_weight + self.ma_weight + self.em_weight + self.lm_weight

        raw_score = (
            self.dc_weight * dc_score
            + self.ma_weight * agreement_score
            + self.em_weight * edge_magnitude_score
            + self.lm_weight * lm_score
        ) / total_weight

        # Apply conservative discount
        final_score = raw_score * self.conservative_discount

        return round(_clamp(final_score, 0.0, 1.0), 3)

    def compute_confidence_with_edge(
        self,
        features: dict,
        baseline_result: dict,
        ml_result: dict,
        differential: float,
        odds_context: Optional[dict] = None,
    ) -> float:
        """
        Compute confidence score incorporating the actual edge magnitude.

        Args:
            features, baseline_result, ml_result: Standard inputs.
            differential: abs(model_total - market_total).
            odds_context: Optional odds movement context.

        Returns:
            Float in [0, 1].
        """
        base_conf = self.compute_confidence_score(
            features, baseline_result, ml_result, odds_context
        )

        # Factor 3 override with actual edge magnitude
        # Scale: 0 pts → 0.0, 5 pts → 0.5, 10+ pts → 1.0
        edge_magnitude_score = _clamp(abs(differential) / 10.0, 0.0, 1.0)

        # Recompute incorporating actual edge
        total_weight = self.dc_weight + self.ma_weight + self.em_weight + self.lm_weight

        data_completeness = float(features.get("data_completeness", 0.5) or 0.5)
        dc_score = _clamp(data_completeness, 0.0, 1.0)

        # Re-use agreement score from base computation
        baseline_total = _get_float(baseline_result, "baseline_total")
        ml_total = _get_float(ml_result, "ml_ensemble_total")
        ml_model_agreement = _get_float(ml_result, "model_agreement_score", 0.5)

        if baseline_total and ml_total:
            diff = abs(baseline_total - ml_total)
            cross_model_agreement = max(0.0, 1.0 - diff / 10.0)
            agreement_score = (cross_model_agreement + float(ml_model_agreement)) / 2.0
        else:
            agreement_score = float(ml_model_agreement) if ml_model_agreement else 0.5

        lm_score = 0.5
        if odds_context:
            opening = odds_context.get("opening_total")
            current = odds_context.get("current_total")
            model_total_val = baseline_total or ml_total or 145.0
            if opening is not None and current is not None:
                if abs(current - model_total_val) < abs(opening - model_total_val):
                    lm_score = 0.7
                elif abs(current - model_total_val) > abs(opening - model_total_val):
                    lm_score = 0.3

        raw_score = (
            self.dc_weight * dc_score
            + self.ma_weight * agreement_score
            + self.em_weight * edge_magnitude_score
            + self.lm_weight * lm_score
        ) / total_weight

        final_score = raw_score * self.conservative_discount
        return round(_clamp(final_score, 0.0, 1.0), 3)

    # ── Confidence interpretation ─────────────────────────────────────────

    @staticmethod
    def interpret_confidence(score: float) -> str:
        """
        Convert a confidence score to a human-readable label.

        Args:
            score: Float in [0, 1].

        Returns:
            'Very High' | 'High' | 'Medium' | 'Low'
        """
        if score >= 0.75:
            return "Very High"
        elif score >= 0.55:
            return "High"
        elif score >= 0.35:
            return "Medium"
        else:
            return "Low"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_float(d: dict, key: str, default: Optional[float] = None) -> Optional[float]:
    v = d.get(key)
    if v is None:
        return default
    try:
        f = float(v)
        if math.isnan(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

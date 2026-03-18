"""
Tests for baseline model, ensemble model, and edge calculator.
"""

from __future__ import annotations

import pytest

from src.models.baseline_model import BaselineModel
from src.models.ensemble import EnsembleModel
from src.pipeline.edge_calculator import EdgeCalculator, format_differential


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def config():
    return {
        "model": {
            "league_avg_possessions": 68.5,
            "league_avg_points_per_100": 105.0,
            "baseline_weight": 0.35,
            "ml_weight": 0.65,
        },
        "confidence": {
            "data_completeness_weight": 0.25,
            "model_agreement_weight": 0.35,
            "edge_magnitude_weight": 0.25,
            "line_movement_weight": 0.15,
            "conservative_discount": 0.85,
        },
        "output": {
            "top_n_console": 20,
            "min_edge_display": 0.0,
            "sort_by": "abs_differential",
        },
    }


@pytest.fixture
def baseline_model(config):
    return BaselineModel(config)


@pytest.fixture
def ensemble_model(config):
    return EnsembleModel(config)


@pytest.fixture
def edge_calculator(config):
    return EdgeCalculator(config)


@pytest.fixture
def typical_features():
    """A realistic feature dict for a D1 game."""
    return {
        "adj_oe_home": 115.0,
        "adj_de_home": 98.0,
        "adj_oe_away": 108.0,
        "adj_de_away": 103.0,
        "home_adj_tempo": 71.0,
        "away_adj_tempo": 68.0,
        "neutral_site": 0,
        "tov_rate_home": 0.16,
        "tov_rate_away": 0.19,
        "ft_rate_home": 0.28,
        "ft_rate_away": 0.31,
        "expected_possessions": 69.5,
        "data_completeness": 0.85,
        "efg_pct_home": 0.545,
        "efg_pct_away": 0.510,
        "orb_rate_home": 0.315,
        "drb_rate_away": 0.701,
        "orb_rate_away": 0.295,
        "drb_rate_home": 0.685,
        "three_pa_rate_home": 0.39,
        "three_pa_rate_away": 0.36,
        "sos_home": 12.1,
        "sos_away": 7.3,
        "conference_home_enc": 1,
        "conference_away_enc": 4,
        "is_conference_game": 0,
        "days_rest_home": 2,
        "days_rest_away": 2,
        "rest_differential": 0,
        "home_court_advantage_adj": 3.5,
        "home_last_5_pts": 74.2,
        "home_last_5_opp_pts": 64.1,
        "home_last_5_total": 138.3,
        "away_last_5_pts": 70.5,
        "away_last_5_opp_pts": 71.2,
        "away_last_5_total": 141.7,
    }


@pytest.fixture
def neutral_features(typical_features):
    return {**typical_features, "neutral_site": 1, "home_court_advantage_adj": 0.0}


@pytest.fixture
def typical_baseline_result():
    return {
        "baseline_total": 148.5,
        "predicted_home": 78.2,
        "predicted_away": 70.3,
        "predicted_possessions": 69.5,
        "baseline_confidence": 0.75,
        "home_ppp": 1.125,
        "away_ppp": 1.012,
        "neutral_site": False,
    }


@pytest.fixture
def typical_ml_result():
    return {
        "xgb_total": 147.2,
        "lgb_total": 149.8,
        "ridge_total": 150.1,
        "ml_ensemble_total": 148.6,
        "model_agreement_score": 0.82,
    }


# ── Baseline Model Tests ──────────────────────────────────────────────────────

class TestBaselineModel:
    def test_predict_returns_dict(self, baseline_model, typical_features):
        result = baseline_model.predict(typical_features)
        assert isinstance(result, dict)

    def test_predict_returns_required_keys(self, baseline_model, typical_features):
        result = baseline_model.predict(typical_features)
        required = [
            "baseline_total",
            "predicted_home",
            "predicted_away",
            "predicted_possessions",
            "baseline_confidence",
        ]
        for key in required:
            assert key in result, f"Missing key: {key}"

    def test_baseline_model_total_is_sum_of_scores(self, baseline_model, typical_features):
        """baseline_total should equal predicted_home + predicted_away."""
        result = baseline_model.predict(typical_features)
        total = result["baseline_total"]
        home = result["predicted_home"]
        away = result["predicted_away"]
        assert abs(total - (home + away)) < 0.1, (
            f"total ({total}) ≠ home ({home}) + away ({away})"
        )

    def test_baseline_model_neutral_site(
        self, baseline_model, typical_features, neutral_features
    ):
        """Neutral site should produce different scores than home game."""
        home_game_result = baseline_model.predict(typical_features)
        neutral_result = baseline_model.predict(neutral_features)

        # Home team should score more at home than neutral
        assert home_game_result["predicted_home"] > neutral_result["predicted_home"], (
            "Home team should score more at home court"
        )
        # Away team should score less at opponent's home
        assert home_game_result["predicted_away"] < neutral_result["predicted_away"], (
            "Away team should score less on road vs neutral"
        )

    def test_baseline_total_in_realistic_range(self, baseline_model, typical_features):
        """Projected total should be within a realistic NCAA range."""
        result = baseline_model.predict(typical_features)
        total = result["baseline_total"]
        assert 100.0 <= total <= 200.0, f"Total {total} outside realistic range [100, 200]"

    def test_possessions_in_realistic_range(self, baseline_model, typical_features):
        """Projected possessions should be in a realistic range."""
        result = baseline_model.predict(typical_features)
        poss = result["predicted_possessions"]
        assert 55.0 <= poss <= 85.0, f"Possessions {poss} outside range [55, 85]"

    def test_confidence_in_range(self, baseline_model, typical_features):
        """Baseline confidence should be in [0, 1]."""
        result = baseline_model.predict(typical_features)
        conf = result["baseline_confidence"]
        assert 0.0 <= conf <= 1.0, f"Confidence {conf} outside [0, 1]"

    def test_predict_with_empty_features(self, baseline_model):
        """Empty feature dict should still produce a result (uses defaults)."""
        result = baseline_model.predict({})
        assert "baseline_total" in result
        total = result["baseline_total"]
        # With all defaults, total should be near 2 * league_avg_ppp * league_avg_poss
        # ≈ 2 * 1.05 * 68.5 ≈ 144
        assert 100 <= total <= 200

    def test_high_tempo_produces_higher_total(self, baseline_model, typical_features):
        """High tempo game should project higher total than slow game."""
        fast_features = {**typical_features, "home_adj_tempo": 78.0, "away_adj_tempo": 76.0}
        slow_features = {**typical_features, "home_adj_tempo": 60.0, "away_adj_tempo": 62.0}

        fast_result = baseline_model.predict(fast_features)
        slow_result = baseline_model.predict(slow_features)

        assert fast_result["baseline_total"] > slow_result["baseline_total"], (
            "High-tempo game should project higher total"
        )


# ── Ensemble Model Tests ──────────────────────────────────────────────────────

class TestEnsembleModel:
    def test_ensemble_predict_returns_dict(
        self, ensemble_model, typical_baseline_result, typical_ml_result
    ):
        result = ensemble_model.predict(typical_baseline_result, typical_ml_result)
        assert isinstance(result, dict)

    def test_ensemble_predict_required_keys(
        self, ensemble_model, typical_baseline_result, typical_ml_result
    ):
        result = ensemble_model.predict(typical_baseline_result, typical_ml_result)
        required = [
            "ensemble_total", "predicted_home_score", "predicted_away_score",
            "predicted_possessions", "blend_mode",
        ]
        for key in required:
            assert key in result, f"Missing ensemble output key: {key}"

    def test_ensemble_total_is_weighted_blend(
        self, ensemble_model, typical_baseline_result, typical_ml_result
    ):
        """Ensemble total should be between baseline and ML totals."""
        result = ensemble_model.predict(typical_baseline_result, typical_ml_result)
        baseline = typical_baseline_result["baseline_total"]
        ml = typical_ml_result["ml_ensemble_total"]
        lo = min(baseline, ml) - 1.0  # Small tolerance
        hi = max(baseline, ml) + 1.0
        ensemble = result["ensemble_total"]
        assert lo <= ensemble <= hi, (
            f"Ensemble {ensemble:.1f} not between baseline {baseline} and ML {ml}"
        )

    def test_ensemble_baseline_only_when_no_ml(
        self, ensemble_model, typical_baseline_result
    ):
        """With empty ML result, ensemble should use baseline only."""
        result = ensemble_model.predict(typical_baseline_result, {})
        assert result["blend_mode"] == "baseline_only"
        assert abs(result["ensemble_total"] - typical_baseline_result["baseline_total"]) < 0.1

    def test_ensemble_conservative_on_large_disagreement(
        self, ensemble_model, typical_baseline_result
    ):
        """Large baseline-ML disagreement should trigger conservative blend."""
        far_ml_result = {
            "ml_ensemble_total": typical_baseline_result["baseline_total"] + 15.0,
            "model_agreement_score": 0.6,
        }
        result = ensemble_model.predict(typical_baseline_result, far_ml_result)
        assert result["blend_mode"] == "conservative"

    def test_ensemble_total_clamped(
        self, ensemble_model, typical_baseline_result, typical_ml_result
    ):
        """Ensemble total should never be unrealistically extreme."""
        result = ensemble_model.predict(typical_baseline_result, typical_ml_result)
        assert 100.0 <= result["ensemble_total"] <= 210.0

    def test_confidence_score_range(
        self, ensemble_model, typical_features, typical_baseline_result, typical_ml_result
    ):
        """Confidence score must be in [0, 1]."""
        conf = ensemble_model.compute_confidence_score(
            typical_features, typical_baseline_result, typical_ml_result
        )
        assert 0.0 <= conf <= 1.0, f"Confidence {conf} out of [0, 1]"

    def test_confidence_higher_with_more_data(
        self, ensemble_model, typical_baseline_result, typical_ml_result
    ):
        """More complete data → higher confidence."""
        low_data = {"data_completeness": 0.2}
        high_data = {"data_completeness": 0.95}

        conf_low = ensemble_model.compute_confidence_score(
            low_data, typical_baseline_result, typical_ml_result
        )
        conf_high = ensemble_model.compute_confidence_score(
            high_data, typical_baseline_result, typical_ml_result
        )
        assert conf_high > conf_low

    def test_interpret_confidence_labels(self, ensemble_model):
        """Interpret confidence should return correct labels for all ranges."""
        assert ensemble_model.interpret_confidence(0.80) == "Very High"
        assert ensemble_model.interpret_confidence(0.60) == "High"
        assert ensemble_model.interpret_confidence(0.40) == "Medium"
        assert ensemble_model.interpret_confidence(0.20) == "Low"
        assert ensemble_model.interpret_confidence(0.0) == "Low"
        assert ensemble_model.interpret_confidence(1.0) == "Very High"


# ── Edge Calculator Tests ─────────────────────────────────────────────────────

class TestEdgeCalculator:
    def test_compute_edge_returns_dict(self, edge_calculator, typical_baseline_result):
        result = edge_calculator.compute_edge(
            {"ensemble_total": 148.5, "game_id": "test_001"},
            market_total=145.0,
        )
        assert isinstance(result, dict)

    def test_edge_calculator_differential_sign_positive(self, edge_calculator):
        """Model > market → positive differential → OVER."""
        result = edge_calculator.compute_edge(
            {"ensemble_total": 150.0}, market_total=145.0
        )
        assert result["differential"] > 0
        assert result["differential"] == pytest.approx(5.0)
        assert result["edge_side"] == "OVER"

    def test_edge_calculator_differential_sign_negative(self, edge_calculator):
        """Model < market → negative differential → UNDER."""
        result = edge_calculator.compute_edge(
            {"ensemble_total": 140.0}, market_total=145.0
        )
        assert result["differential"] < 0
        assert result["differential"] == pytest.approx(-5.0)
        assert result["edge_side"] == "UNDER"

    def test_edge_calculator_over_under_labeling(self, edge_calculator):
        """OVER/UNDER labeling correctness."""
        over = edge_calculator.compute_edge({"ensemble_total": 152.0}, 148.5)
        under = edge_calculator.compute_edge({"ensemble_total": 143.0}, 148.5)
        push = edge_calculator.compute_edge({"ensemble_total": 148.5}, 148.5)

        assert over["edge_side"] == "OVER"
        assert under["edge_side"] == "UNDER"
        assert push["edge_side"] == "PUSH"

    def test_abs_differential(self, edge_calculator):
        """abs_differential should always be non-negative."""
        result = edge_calculator.compute_edge({"ensemble_total": 140.0}, 145.0)
        assert result["abs_differential"] == 5.0
        assert result["abs_differential"] >= 0

    def test_edge_bucket_assignment(self, edge_calculator):
        """Each differential range should map to the correct bucket."""
        assert edge_calculator.compute_edge({"ensemble_total": 146.5}, 145.0)["edge_bucket"] == "0-2"
        assert edge_calculator.compute_edge({"ensemble_total": 148.0}, 145.0)["edge_bucket"] == "2-4"
        assert edge_calculator.compute_edge({"ensemble_total": 150.0}, 145.0)["edge_bucket"] == "4-6"
        assert edge_calculator.compute_edge({"ensemble_total": 153.0}, 145.0)["edge_bucket"] == "6-8"
        assert edge_calculator.compute_edge({"ensemble_total": 155.0}, 145.0)["edge_bucket"] == "8+"

    def test_rank_edges_sorts_by_abs_differential(self, edge_calculator):
        """rank_edges should sort by abs_differential descending."""
        edges = [
            edge_calculator.compute_edge({"ensemble_total": 147.0, "game_id": "g1"}, 145.0),
            edge_calculator.compute_edge({"ensemble_total": 155.0, "game_id": "g2"}, 145.0),
            edge_calculator.compute_edge({"ensemble_total": 141.0, "game_id": "g3"}, 145.0),
        ]
        ranked = edge_calculator.rank_edges(edges)

        assert len(ranked) == 3
        diffs = ranked["abs_differential"].tolist()
        assert diffs == sorted(diffs, reverse=True), "Not sorted by abs_differential desc"

    def test_rank_edges_adds_rank_column(self, edge_calculator):
        """rank_edges should add a 'rank' column starting at 1."""
        edges = [
            edge_calculator.compute_edge({"ensemble_total": 148.0, "game_id": f"g{i}"}, 145.0)
            for i in range(3)
        ]
        ranked = edge_calculator.rank_edges(edges)
        assert "rank" in ranked.columns
        assert ranked["rank"].iloc[0] == 1

    def test_rank_edges_empty_input(self, edge_calculator):
        """rank_edges with empty list should return empty DataFrame."""
        import pandas as pd
        result = edge_calculator.rank_edges([])
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_format_differential(self):
        """format_differential should produce sign-prefixed strings."""
        assert format_differential(3.5) == "+3.5"
        assert format_differential(-3.5) == "-3.5"
        assert format_differential(0.0) == "0.0"
        assert format_differential(None) == "—"

    def test_format_differential_large_values(self):
        """Should handle larger values without error."""
        assert format_differential(12.0) == "+12.0"
        assert format_differential(-10.5) == "-10.5"

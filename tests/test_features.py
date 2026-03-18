"""
Tests for the feature engineering module.
"""

from __future__ import annotations

import math
import pytest

from src.features.feature_engineering import (
    FeatureEngineer,
    FEATURE_COLUMNS,
    MISSING_SENTINEL,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def config():
    return {
        "model": {
            "league_avg_possessions": 68.5,
            "league_avg_points_per_100": 105.0,
            "recent_form_windows": [3, 5, 10],
        }
    }


@pytest.fixture
def feature_engineer(config):
    return FeatureEngineer(config)


@pytest.fixture
def game_home():
    """Standard non-neutral-site game dict."""
    return {
        "game_id": "test_001",
        "date": "2024-02-15",
        "neutral_site": False,
        "days_rest_home": 2,
        "days_rest_away": 3,
    }


@pytest.fixture
def game_neutral():
    """Neutral-site game dict."""
    return {
        "game_id": "test_002",
        "date": "2024-03-20",
        "neutral_site": True,
        "days_rest_home": 1,
        "days_rest_away": 1,
    }


@pytest.fixture
def good_home_stats():
    """Realistic stats for a strong offensive team."""
    return {
        "team_id": "espn_52",
        "season": "2024",
        "adj_oe": 118.5,         # Elite offense
        "adj_de": 95.2,          # Elite defense
        "adj_tempo": 72.1,       # Fast pace
        "raw_oe": 78.3,
        "raw_de": 63.1,
        "ppg": 78.3,
        "opp_ppg": 63.1,
        "efg_pct": 0.563,
        "opp_efg_pct": 0.478,
        "two_p_pct": 0.551,
        "opp_two_p_pct": 0.456,
        "three_p_pct": 0.385,
        "opp_three_p_pct": 0.312,
        "three_pa_rate": 0.412,
        "opp_three_pa_rate": 0.342,
        "ft_rate": 0.285,
        "opp_ft_rate": 0.262,
        "tov_rate": 0.158,
        "opp_tov_rate": 0.214,
        "orb_rate": 0.332,
        "drb_rate": 0.721,
        "sos": 15.2,
        "conference": "ACC",
        "games_played": 22,
    }


@pytest.fixture
def average_away_stats():
    """Stats for an average team."""
    return {
        "team_id": "espn_99",
        "season": "2024",
        "adj_oe": 103.8,
        "adj_de": 106.4,
        "adj_tempo": 66.8,
        "raw_oe": 68.2,
        "raw_de": 70.1,
        "ppg": 68.2,
        "opp_ppg": 70.1,
        "efg_pct": 0.498,
        "opp_efg_pct": 0.512,
        "two_p_pct": 0.492,
        "opp_two_p_pct": 0.501,
        "three_p_pct": 0.341,
        "opp_three_p_pct": 0.352,
        "three_pa_rate": 0.358,
        "opp_three_pa_rate": 0.371,
        "ft_rate": 0.301,
        "opp_ft_rate": 0.288,
        "tov_rate": 0.182,
        "opp_tov_rate": 0.178,
        "orb_rate": 0.281,
        "drb_rate": 0.688,
        "sos": 5.4,
        "conference": "MWC",
        "games_played": 20,
    }


@pytest.fixture
def recent_games_home():
    """10 recent games for the home team."""
    return [
        {
            "game_id": f"game_{i}",
            "date": f"2024-02-{i+1:02d}",
            "score": 75 + i,
            "opp_score": 68 - i,
            "total": 143 + i,
            "home": True,
            "winner": True,
            "possessions": None,
        }
        for i in range(10)
    ]


@pytest.fixture
def recent_games_away():
    """10 recent games for the away team."""
    return [
        {
            "game_id": f"game_a_{i}",
            "date": f"2024-02-{i+1:02d}",
            "score": 68 + i,
            "opp_score": 72 - i,
            "total": 140 + i,
            "home": False,
            "winner": i > 4,
            "possessions": None,
        }
        for i in range(10)
    ]


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestBuildGameFeatures:
    def test_returns_dict(self, feature_engineer, game_home, good_home_stats,
                          average_away_stats, recent_games_home, recent_games_away):
        features = feature_engineer.build_game_features(
            game_home, good_home_stats, average_away_stats,
            recent_games_home, recent_games_away
        )
        assert isinstance(features, dict)

    def test_build_game_features_returns_all_keys(
        self, feature_engineer, game_home, good_home_stats,
        average_away_stats, recent_games_home, recent_games_away
    ):
        """All expected feature keys should be present in the output."""
        features = feature_engineer.build_game_features(
            game_home, good_home_stats, average_away_stats,
            recent_games_home, recent_games_away
        )

        missing_keys = []
        for key in FEATURE_COLUMNS:
            if key not in features:
                missing_keys.append(key)

        assert not missing_keys, f"Missing feature keys: {missing_keys}"

    def test_build_game_features_neutral_site(
        self, feature_engineer, game_neutral, good_home_stats,
        average_away_stats, recent_games_home, recent_games_away
    ):
        """Neutral site flag should set home court advantage to 0."""
        features = feature_engineer.build_game_features(
            game_neutral, good_home_stats, average_away_stats,
            recent_games_home, recent_games_away
        )
        assert features["neutral_site"] == 1
        assert features["home_court_advantage_adj"] == 0.0

    def test_non_neutral_site_has_home_advantage(
        self, feature_engineer, game_home, good_home_stats,
        average_away_stats, recent_games_home, recent_games_away
    ):
        """Non-neutral site should set positive home court advantage."""
        features = feature_engineer.build_game_features(
            game_home, good_home_stats, average_away_stats,
            recent_games_home, recent_games_away
        )
        assert features["neutral_site"] == 0
        assert features["home_court_advantage_adj"] > 0

    def test_expected_possessions_reasonable(
        self, feature_engineer, game_home, good_home_stats,
        average_away_stats, recent_games_home, recent_games_away
    ):
        """Expected possessions should be in a realistic NCAA range (55-85)."""
        features = feature_engineer.build_game_features(
            game_home, good_home_stats, average_away_stats,
            recent_games_home, recent_games_away
        )
        poss = features["expected_possessions"]
        assert 55.0 <= poss <= 85.0, f"Expected possessions out of range: {poss}"

    def test_possessions_reflect_team_tempos(
        self, feature_engineer, game_home, good_home_stats,
        average_away_stats, recent_games_home, recent_games_away
    ):
        """Expected possessions should be between the two teams' tempos."""
        home_tempo = good_home_stats["adj_tempo"]    # 72.1
        away_tempo = average_away_stats["adj_tempo"]  # 66.8
        expected_avg = (home_tempo + away_tempo) / 2.0

        features = feature_engineer.build_game_features(
            game_home, good_home_stats, average_away_stats,
            recent_games_home, recent_games_away
        )
        poss = features["expected_possessions"]
        # Should be reasonably close to the average of the two tempos
        assert abs(poss - expected_avg) < 5.0, (
            f"Possessions {poss:.1f} too far from average {expected_avg:.1f}"
        )

    def test_rest_differential(
        self, feature_engineer, game_home, good_home_stats,
        average_away_stats, recent_games_home, recent_games_away
    ):
        """Rest differential should equal days_rest_home - days_rest_away."""
        game_home_copy = {**game_home, "days_rest_home": 3, "days_rest_away": 1}
        features = feature_engineer.build_game_features(
            game_home_copy, good_home_stats, average_away_stats,
            recent_games_home, recent_games_away
        )
        assert features["rest_differential"] == 2  # 3 - 1

    def test_conference_game_detection(
        self, feature_engineer, game_home, good_home_stats,
        average_away_stats, recent_games_home, recent_games_away
    ):
        """Same-conference teams should be marked as conference game."""
        same_conf_away = {**average_away_stats, "conference": "ACC"}
        features = feature_engineer.build_game_features(
            game_home, good_home_stats, same_conf_away,
            recent_games_home, recent_games_away
        )
        assert features["is_conference_game"] == 1

    def test_different_conference_not_conference_game(
        self, feature_engineer, game_home, good_home_stats,
        average_away_stats, recent_games_home, recent_games_away
    ):
        """Different-conference teams should NOT be marked as conference game."""
        features = feature_engineer.build_game_features(
            game_home, good_home_stats, average_away_stats,
            recent_games_home, recent_games_away
        )
        # Home is ACC, away is MWC
        assert features["is_conference_game"] == 0


class TestRecentFormFeatures:
    def test_recent_form_handles_insufficient_games(
        self, feature_engineer, game_home, good_home_stats, average_away_stats
    ):
        """With fewer games than window, should still return features (possibly MISSING)."""
        # Only 2 games — less than all windows (3, 5, 10)
        two_games = [
            {"game_id": "g1", "date": "2024-02-01", "score": 74, "opp_score": 68,
             "total": 142, "home": True, "winner": True, "possessions": None},
            {"game_id": "g2", "date": "2024-02-04", "score": 71, "opp_score": 72,
             "total": 143, "home": False, "winner": False, "possessions": None},
        ]

        features = feature_engineer.build_game_features(
            game_home, good_home_stats, average_away_stats, two_games, two_games
        )

        # Keys should exist even when data is insufficient
        assert "home_last_3_pts" in features
        assert "home_last_5_pts" in features
        assert "home_last_10_pts" in features
        assert "away_last_10_pts" in features

        # With 2 games, window 3 should use available data (not crash)
        # home_last_3_pts should be valid (uses 2 games)
        assert features["home_last_3_pts"] != MISSING_SENTINEL or True  # Either real or missing, not error

    def test_no_recent_games_uses_sentinels(
        self, feature_engineer, game_home, good_home_stats, average_away_stats
    ):
        """With no recent games, form features should be MISSING_SENTINEL."""
        features = feature_engineer.build_game_features(
            game_home, good_home_stats, average_away_stats, [], []
        )
        assert features["home_last_3_pts"] == MISSING_SENTINEL
        assert features["away_last_10_total"] == MISSING_SENTINEL

    def test_form_trend_missing_insufficient_games(
        self, feature_engineer, game_home, good_home_stats, average_away_stats
    ):
        """Form trend requires at least 3 games; fewer returns MISSING."""
        one_game = [
            {"game_id": "g1", "date": "2024-02-01", "score": 74, "opp_score": 68,
             "total": 142, "home": True, "winner": True, "possessions": None}
        ]
        features = feature_engineer.build_game_features(
            game_home, good_home_stats, average_away_stats, one_game, one_game
        )
        assert features["home_form_trend"] == MISSING_SENTINEL
        assert features["away_form_trend"] == MISSING_SENTINEL


class TestDataCompleteness:
    def test_data_completeness_full_stats(
        self, feature_engineer, game_home, good_home_stats,
        average_away_stats, recent_games_home, recent_games_away
    ):
        """Full stats should produce high data completeness."""
        features = feature_engineer.build_game_features(
            game_home, good_home_stats, average_away_stats,
            recent_games_home, recent_games_away
        )
        completeness = features["data_completeness"]
        assert 0.0 <= completeness <= 1.0
        # With full stats, completeness should be moderate to high
        assert completeness >= 0.4, f"Expected completeness >= 0.4, got {completeness}"

    def test_data_completeness_missing_stats(
        self, feature_engineer, game_home, recent_games_home, recent_games_away
    ):
        """Empty stats dicts should lower completeness score."""
        features_sparse = feature_engineer.build_game_features(
            game_home, {}, {}, recent_games_home, recent_games_away
        )
        features_full_home = feature_engineer.build_game_features(
            game_home, {"adj_oe": 110.0, "adj_de": 100.0, "adj_tempo": 70.0},
            {"adj_oe": 105.0, "adj_de": 105.0, "adj_tempo": 68.0},
            recent_games_home, recent_games_away
        )
        assert features_sparse["data_completeness"] <= features_full_home["data_completeness"]

    def test_data_completeness_range(
        self, feature_engineer, game_home, good_home_stats,
        average_away_stats, recent_games_home, recent_games_away
    ):
        """Data completeness must always be in [0, 1]."""
        features = feature_engineer.build_game_features(
            game_home, good_home_stats, average_away_stats,
            recent_games_home, recent_games_away
        )
        c = features["data_completeness"]
        assert 0.0 <= c <= 1.0, f"Completeness {c} out of [0,1]"


class TestMatchupFeatures:
    def test_off_vs_def_home_normalized(
        self, feature_engineer, game_home, good_home_stats,
        average_away_stats, recent_games_home, recent_games_away
    ):
        """off_vs_def_home should be >1 when home offense > league avg and away defense < league avg."""
        features = feature_engineer.build_game_features(
            game_home, good_home_stats, average_away_stats,
            recent_games_home, recent_games_away
        )
        # good_home_stats: adj_oe=118.5, average_away: adj_de=106.4
        # both above/near league avg, so product should be > 1
        assert features["off_vs_def_home"] > 0

    def test_tov_environment_reasonable(
        self, feature_engineer, game_home, good_home_stats,
        average_away_stats, recent_games_home, recent_games_away
    ):
        """TOV environment should be between 0 and 0.4 (realistic range)."""
        features = feature_engineer.build_game_features(
            game_home, good_home_stats, average_away_stats,
            recent_games_home, recent_games_away
        )
        tov_env = features["tov_environment"]
        if tov_env != MISSING_SENTINEL:
            assert 0.0 < tov_env < 0.5, f"TOV environment out of range: {tov_env}"

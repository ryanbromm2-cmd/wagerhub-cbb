"""
Demo data for WagerHub dashboard.

Shown automatically when the pipeline hasn't been run yet so the dashboard
looks fully functional on first launch. Every value is realistic NCAA data.
"""
from __future__ import annotations
from datetime import datetime, date

# ── Today's demo slate ────────────────────────────────────────────────────────

def get_demo_slate(target_date: str | None = None) -> list[dict]:
    """Return a realistic demo slate for the given date."""
    d = target_date or date.today().strftime("%Y-%m-%d")
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    games = [
        # Strong UNDER — the user's flagship example
        dict(
            game_id="demo_mia_smu",
            game_date=d,
            away_team="Miami (OH)",
            home_team="SMU",
            market_total=163.5,
            ensemble_total=150.0,
            baseline_total=149.2,
            ml_total=150.6,
            differential=-13.5,
            abs_differential=13.5,
            edge_side="UNDER",
            confidence_score=0.88,
            predicted_possessions=67.8,
            predicted_away_score=74.9,
            predicted_home_score=75.1,
            sportsbook="DraftKings",
            over_odds=-110,
            under_odds=-110,
            run_timestamp=ts,
            game_time="8:00 PM ET",
            status="scheduled",
        ),
        # Strong OVER — rivalry game, high tempo
        dict(
            game_id="demo_duke_unc",
            game_date=d,
            away_team="Duke",
            home_team="North Carolina",
            market_total=157.5,
            ensemble_total=164.2,
            baseline_total=163.5,
            ml_total=164.8,
            differential=6.7,
            abs_differential=6.7,
            edge_side="OVER",
            confidence_score=0.82,
            predicted_possessions=72.1,
            predicted_away_score=82.3,
            predicted_home_score=81.9,
            sportsbook="DraftKings",
            over_odds=-115,
            under_odds=-105,
            run_timestamp=ts,
            game_time="9:00 PM ET",
            status="scheduled",
        ),
        # Strong OVER — high-volume teams
        dict(
            game_id="demo_uconn_prov",
            game_date=d,
            away_team="UConn",
            home_team="Providence",
            market_total=136.0,
            ensemble_total=142.5,
            baseline_total=141.8,
            ml_total=143.1,
            differential=6.5,
            abs_differential=6.5,
            edge_side="OVER",
            confidence_score=0.76,
            predicted_possessions=68.4,
            predicted_away_score=71.9,
            predicted_home_score=70.6,
            sportsbook="FanDuel",
            over_odds=-112,
            under_odds=-108,
            run_timestamp=ts,
            game_time="6:30 PM ET",
            status="scheduled",
        ),
        # Medium UNDER
        dict(
            game_id="demo_gonz_stm",
            game_date=d,
            away_team="Gonzaga",
            home_team="Saint Mary's",
            market_total=149.0,
            ensemble_total=143.8,
            baseline_total=144.5,
            ml_total=143.2,
            differential=-5.2,
            abs_differential=5.2,
            edge_side="UNDER",
            confidence_score=0.71,
            predicted_possessions=65.2,
            predicted_away_score=72.4,
            predicted_home_score=71.4,
            sportsbook="BetMGM",
            over_odds=-108,
            under_odds=-112,
            run_timestamp=ts,
            game_time="10:00 PM ET",
            status="scheduled",
        ),
        # Medium OVER
        dict(
            game_id="demo_mich_purd",
            game_date=d,
            away_team="Michigan St.",
            home_team="Purdue",
            market_total=137.0,
            ensemble_total=141.8,
            baseline_total=142.0,
            ml_total=141.6,
            differential=4.8,
            abs_differential=4.8,
            edge_side="OVER",
            confidence_score=0.65,
            predicted_possessions=66.9,
            predicted_away_score=70.5,
            predicted_home_score=71.3,
            sportsbook="DraftKings",
            over_odds=-110,
            under_odds=-110,
            run_timestamp=ts,
            game_time="7:00 PM ET",
            status="scheduled",
        ),
        # Medium UNDER
        dict(
            game_id="demo_tenn_uk",
            game_date=d,
            away_team="Tennessee",
            home_team="Kentucky",
            market_total=131.5,
            ensemble_total=127.2,
            baseline_total=126.8,
            ml_total=127.5,
            differential=-4.3,
            abs_differential=4.3,
            edge_side="UNDER",
            confidence_score=0.68,
            predicted_possessions=62.1,
            predicted_away_score=63.9,
            predicted_home_score=63.3,
            sportsbook="Pinnacle",
            over_odds=-105,
            under_odds=-115,
            run_timestamp=ts,
            game_time="8:30 PM ET",
            status="scheduled",
        ),
        # Medium OVER
        dict(
            game_id="demo_marq_crei",
            game_date=d,
            away_team="Marquette",
            home_team="Creighton",
            market_total=151.0,
            ensemble_total=155.4,
            baseline_total=154.9,
            ml_total=155.8,
            differential=4.4,
            abs_differential=4.4,
            edge_side="OVER",
            confidence_score=0.62,
            predicted_possessions=70.3,
            predicted_away_score=77.5,
            predicted_home_score=77.9,
            sportsbook="FanDuel",
            over_odds=-110,
            under_odds=-110,
            run_timestamp=ts,
            game_time="7:00 PM ET",
            status="scheduled",
        ),
        # Medium OVER
        dict(
            game_id="demo_kans_bay",
            game_date=d,
            away_team="Kansas",
            home_team="Baylor",
            market_total=145.0,
            ensemble_total=148.8,
            baseline_total=149.1,
            ml_total=148.5,
            differential=3.8,
            abs_differential=3.8,
            edge_side="OVER",
            confidence_score=0.58,
            predicted_possessions=69.5,
            predicted_away_score=74.8,
            predicted_home_score=74.0,
            sportsbook="DraftKings",
            over_odds=-110,
            under_odds=-110,
            run_timestamp=ts,
            game_time="9:00 PM ET",
            status="scheduled",
        ),
        # Medium UNDER
        dict(
            game_id="demo_iowa_tt",
            game_date=d,
            away_team="Iowa St.",
            home_team="Texas Tech",
            market_total=140.5,
            ensemble_total=137.0,
            baseline_total=136.5,
            ml_total=137.4,
            differential=-3.5,
            abs_differential=3.5,
            edge_side="UNDER",
            confidence_score=0.55,
            predicted_possessions=65.8,
            predicted_away_score=68.8,
            predicted_home_score=68.2,
            sportsbook="BetMGM",
            over_odds=-108,
            under_odds=-112,
            run_timestamp=ts,
            game_time="6:00 PM ET",
            status="scheduled",
        ),
        # Mild OVER
        dict(
            game_id="demo_aub_ala",
            game_date=d,
            away_team="Auburn",
            home_team="Alabama",
            market_total=155.5,
            ensemble_total=158.9,
            baseline_total=158.2,
            ml_total=159.5,
            differential=3.4,
            abs_differential=3.4,
            edge_side="OVER",
            confidence_score=0.51,
            predicted_possessions=71.2,
            predicted_away_score=79.3,
            predicted_home_score=79.6,
            sportsbook="Pinnacle",
            over_odds=-110,
            under_odds=-110,
            run_timestamp=ts,
            game_time="8:00 PM ET",
            status="scheduled",
        ),
        # Mild OVER
        dict(
            game_id="demo_ucla_az",
            game_date=d,
            away_team="UCLA",
            home_team="Arizona",
            market_total=148.5,
            ensemble_total=151.0,
            baseline_total=150.4,
            ml_total=151.5,
            differential=2.5,
            abs_differential=2.5,
            edge_side="OVER",
            confidence_score=0.42,
            predicted_possessions=68.8,
            predicted_away_score=75.1,
            predicted_home_score=75.9,
            sportsbook="FanDuel",
            over_odds=-112,
            under_odds=-108,
            run_timestamp=ts,
            game_time="10:30 PM ET",
            status="scheduled",
        ),
        # Mild UNDER
        dict(
            game_id="demo_ill_wis",
            game_date=d,
            away_team="Illinois",
            home_team="Wisconsin",
            market_total=132.5,
            ensemble_total=130.1,
            baseline_total=130.5,
            ml_total=129.8,
            differential=-2.4,
            abs_differential=2.4,
            edge_side="UNDER",
            confidence_score=0.40,
            predicted_possessions=61.5,
            predicted_away_score=65.1,
            predicted_home_score=65.0,
            sportsbook="DraftKings",
            over_odds=-110,
            under_odds=-110,
            run_timestamp=ts,
            game_time="7:30 PM ET",
            status="scheduled",
        ),
    ]

    # Sort by abs_differential descending (biggest edge first)
    return sorted(games, key=lambda g: g["abs_differential"], reverse=True)


# ── Demo line history ─────────────────────────────────────────────────────────

def get_demo_line_history(game_id: str) -> list[dict]:
    """Return fake line movement history for a given demo game."""
    import random
    from datetime import timedelta

    histories = {
        "demo_mia_smu": [
            # Line opened at 160, moved up to 163.5
            (0,   "DraftKings", 160.0),
            (4,   "DraftKings", 160.5),
            (8,   "DraftKings", 161.0),
            (12,  "DraftKings", 162.0),
            (18,  "DraftKings", 163.0),
            (22,  "DraftKings", 163.5),
            (0,   "FanDuel",    160.0),
            (6,   "FanDuel",    161.0),
            (14,  "FanDuel",    162.5),
            (22,  "FanDuel",    163.5),
        ],
        "demo_duke_unc": [
            (0,   "DraftKings", 155.0),
            (3,   "DraftKings", 155.5),
            (10,  "DraftKings", 156.0),
            (18,  "DraftKings", 157.0),
            (22,  "DraftKings", 157.5),
            (0,   "FanDuel",    154.5),
            (8,   "FanDuel",    155.5),
            (20,  "FanDuel",    157.0),
        ],
    }

    now = datetime.now()
    base_history = histories.get(game_id, [
        (0,  "DraftKings", 145.0),
        (6,  "DraftKings", 145.5),
        (14, "DraftKings", 146.0),
        (20, "DraftKings", 146.0),
    ])

    result = []
    for hours_ago, book, total in base_history:
        ts = now - timedelta(hours=hours_ago)
        result.append({
            "game_id": game_id,
            "sportsbook": book,
            "total": total,
            "timestamp": ts,
        })

    return sorted(result, key=lambda r: r["timestamp"])


# ── Demo historical / backtest ────────────────────────────────────────────────

def get_demo_historical() -> list[dict]:
    """Return fake historical backtest results for the performance page."""
    import random
    random.seed(42)

    results = []
    edge_buckets = ["0-2", "2-4", "4-6", "6-8", "8+"]
    conferences = ["Big East", "SEC", "Big Ten", "Big 12", "ACC", "Pac-12", "WCC"]
    result_choices_by_bucket = {
        "0-2": ["over", "under", "lose_over", "lose_under"],
        "2-4": ["over", "lose_over", "under", "lose_under"],
        "4-6": ["over", "over", "lose_over", "under"],
        "6-8": ["over", "over", "over", "lose_over"],
        "8+":  ["over", "over", "over", "over", "lose_over"],
    }

    from datetime import timedelta
    base = date(2024, 11, 15)
    game_num = 1
    for i in range(200):
        d = base + timedelta(days=random.randint(0, 120))
        projected = random.uniform(128, 162)
        actual = projected + random.gauss(0, 6)
        market = projected + random.gauss(0, 3)
        diff = projected - market
        edge_side = "OVER" if diff > 0 else "UNDER"
        bucket = (
            "0-2" if abs(diff) < 2 else
            "2-4" if abs(diff) < 4 else
            "4-6" if abs(diff) < 6 else
            "6-8" if abs(diff) < 8 else "8+"
        )
        choices = result_choices_by_bucket[bucket]
        result = random.choice(choices)
        results.append({
            "game_id": f"hist_{i}",
            "game_date": d.strftime("%Y-%m-%d"),
            "projected_total": round(projected, 1),
            "market_total": round(market, 1),
            "actual_total": round(actual, 1),
            "differential": round(diff, 1),
            "edge_side": edge_side,
            "result": result,
            "edge_bucket": bucket,
            "conference": random.choice(conferences),
        })
        game_num += 1

    return sorted(results, key=lambda r: r["game_date"], reverse=True)

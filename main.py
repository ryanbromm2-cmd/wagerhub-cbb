#!/usr/bin/env python3
"""
CBB Totals Model — Main Entry Point

Usage:
  python main.py run                              # Run today's pipeline
  python main.py run --date 2024-02-15            # Run for specific date
  python main.py run --date 2024-02-15 --force    # Force re-fetch stats
  python main.py refresh-odds                     # Refresh odds only (intraday)
  python main.py refresh-odds --date 2024-02-15   # Refresh odds for date
  python main.py train                            # Train/retrain ML models
  python main.py backtest                         # Run backtest (last 90 days)
  python main.py backtest --start 2023-01-01 --end 2024-03-31
  python main.py collect --start 2023-01-01 --end 2024-03-31
  python main.py dashboard                        # Launch Streamlit dashboard
  python main.py schedule                         # Start the scheduler daemon
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

# ── Project root on sys.path ───────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Load .env ──────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

VERSION = "1.0.0"
BANNER = f"""
╔══════════════════════════════════════════════════════════════╗
║          CBB Totals Model  v{VERSION}                            ║
║          NCAA Men's Basketball Over/Under Projection          ║
╚══════════════════════════════════════════════════════════════╝
"""


# ── Config loader ─────────────────────────────────────────────────────────────

def load_config() -> dict:
    """Load master config from config/config.yaml."""
    import yaml
    cfg_path = PROJECT_ROOT / "config" / "config.yaml"
    if not cfg_path.exists():
        print(f"ERROR: config file not found at {cfg_path}")
        sys.exit(1)
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def get_db(config: dict):
    """Initialize and return DatabaseManager."""
    from src.utils.db import DatabaseManager
    db = DatabaseManager(config)
    db.init_db()
    return db


# ── Command handlers ──────────────────────────────────────────────────────────

def cmd_run(args) -> None:
    """Run the full daily pipeline."""
    print(BANNER)
    config = load_config()
    db = get_db(config)

    from src.pipeline.daily_pipeline import DailyPipeline
    pipeline = DailyPipeline(config, db)

    run_date = args.date or date.today().strftime("%Y-%m-%d")
    force = getattr(args, "force", False)

    print(f"Running pipeline for {run_date} (force_refresh={force})")
    result = pipeline.run(run_date=run_date, force_refresh=force)

    print(f"\nPipeline complete.")
    print(f"  Games processed:  {result.get('games_processed', 0)}")
    print(f"  Games with odds:  {result.get('games_with_odds', 0)}")
    print(f"  Run time:         {result.get('run_time_s', 0)}s")
    if result.get("csv_path"):
        print(f"  CSV output:       {result['csv_path']}")

    top = result.get("top_edges", [])
    if top:
        print(f"\nTop Edges:")
        for e in top[:5]:
            home = e.get("home_team_id", e.get("home_team", "?"))
            away = e.get("away_team_id", e.get("away_team", "?"))
            diff = e.get("differential", 0)
            side = e.get("edge_side", "?")
            mkt = e.get("market_total", "?")
            sign = "+" if diff and diff > 0 else ""
            print(f"  {away} @ {home} | Mkt: {mkt} | {side} {sign}{diff:.1f}")


def cmd_refresh_odds(args) -> None:
    """Refresh odds only (intraday)."""
    print(BANNER)
    config = load_config()
    db = get_db(config)

    from src.pipeline.daily_pipeline import DailyPipeline
    pipeline = DailyPipeline(config, db)

    run_date = args.date or date.today().strftime("%Y-%m-%d")
    print(f"Refreshing odds for {run_date}...")
    result = pipeline.refresh_odds_only(run_date=run_date)
    print(f"Refresh complete. {result.get('updated', 0)} games updated in {result.get('run_time_s', 0)}s.")


def cmd_train(args) -> None:
    """Train / retrain the ML models."""
    print(BANNER)
    config = load_config()
    db = get_db(config)

    from src.models.ml_model import MLModelTrainer
    from src.features.feature_engineering import FEATURE_COLUMNS

    print("Gathering training data from database...")

    # Fetch all historical projections with actual results
    import pandas as pd
    hist = db.get_historical_projections("2020-01-01", date.today().strftime("%Y-%m-%d"))
    if not hist:
        print("ERROR: No historical data found. Run 'collect' first to gather game data.")
        return

    hist_df = pd.DataFrame(hist)
    completed = hist_df[hist_df["actual_total"].notna()]

    if len(completed) < 50:
        print(f"ERROR: Only {len(completed)} completed games with projections. Need ≥50.")
        return

    print(f"Building feature matrix from {len(completed)} completed games...")

    # Load features for each game
    rows = []
    targets = []
    for _, row in completed.iterrows():
        gid = row.get("game_id")
        if not gid:
            continue
        features = db.get_game_features(gid)
        if features:
            rows.append(features)
            targets.append(float(row["actual_total"]))

    if len(rows) < 50:
        print(f"ERROR: Only {len(rows)} games with features. Run the pipeline on historical dates.")
        return

    features_df = pd.DataFrame(rows)
    target_series = pd.Series(targets, name="total")

    print(f"Training on {len(features_df)} samples...")

    trainer = MLModelTrainer(config)

    # Cross-validation first
    cv_results = trainer.time_aware_cross_validate(features_df, target_series, n_splits=5)
    print(
        f"CV Results: MAE={cv_results.get('cv_mae_mean'):.2f} "
        f"± {cv_results.get('cv_mae_std'):.2f}"
    )

    # Full training
    trainer.train(features_df, target_series)

    # Save
    models_dir = PROJECT_ROOT / "models"
    save_path = trainer.save_models(str(models_dir))
    print(f"Models saved to {save_path}")

    # Feature importance
    fi = trainer.get_feature_importance()
    if not fi.empty:
        print("\nTop 10 Features:")
        for _, row in fi.head(10).iterrows():
            print(f"  {row['feature']:<40} {row['importance']:.4f}")


def cmd_backtest(args) -> None:
    """Run historical backtest."""
    print(BANNER)
    config = load_config()
    db = get_db(config)

    from src.backtest.backtest import Backtester
    from src.features.feature_engineering import FeatureEngineer
    from src.models.baseline_model import BaselineModel
    from src.models.ensemble import EnsembleModel
    from src.models.ml_model import MLModelTrainer, MLModelPredictor

    start_date = getattr(args, "start", None) or (
        date.today() - timedelta(days=90)
    ).strftime("%Y-%m-%d")
    end_date = getattr(args, "end", None) or date.today().strftime("%Y-%m-%d")

    print(f"Running backtest from {start_date} to {end_date}...")

    feature_engineer = FeatureEngineer(config)
    baseline_model = BaselineModel(config)
    ensemble_model = EnsembleModel(config)

    # Try to load ML models
    ml_predictor = None
    models_dir = PROJECT_ROOT / "models"
    model_file = models_dir / "cbb_totals_models.joblib"
    if model_file.exists():
        try:
            trainer = MLModelTrainer(config)
            trainer.load_models(str(models_dir))
            ml_predictor = MLModelPredictor(trainer)
            print("ML models loaded for backtest.")
        except Exception as e:
            print(f"Could not load ML models ({e}); using baseline only.")

    backtester = Backtester(config)
    report = backtester.run_backtest(
        start_date=start_date,
        end_date=end_date,
        db_manager=db,
        feature_engineer=feature_engineer,
        baseline_model=baseline_model,
        ml_predictor=ml_predictor,
        ensemble_model=ensemble_model,
    )

    report.print_summary()

    csv_path = report.export_csv()
    if csv_path:
        print(f"\nDetailed results saved to: {csv_path}")


def cmd_collect(args) -> None:
    """Collect historical game data and stats."""
    print(BANNER)
    config = load_config()
    db = get_db(config)

    from src.data.base_adapter import DataAdapterFactory
    from src.features.feature_engineering import FeatureEngineer
    from src.pipeline.daily_pipeline import DailyPipeline, _date_to_season

    start = getattr(args, "start", None) or (
        date.today() - timedelta(days=90)
    ).strftime("%Y-%m-%d")
    end = getattr(args, "end", None) or date.today().strftime("%Y-%m-%d")

    print(f"Collecting historical data from {start} to {end}...")

    factory = DataAdapterFactory(config)
    schedule_adapter = factory.get_schedule_adapter()
    stats_adapter = factory.get_stats_adapter()
    feature_engineer = FeatureEngineer(config)

    print(f"Fetching completed games from ESPN ({start} → {end})...")
    games = schedule_adapter.get_completed_games_by_date_range(start, end)
    print(f"Found {len(games)} completed games.")

    for i, game in enumerate(games):
        game_id = game.get("game_id", "")
        game_date = game.get("date", "")

        # Save game
        try:
            db.upsert_game(game)
        except Exception as exc:
            print(f"  Could not save game {game_id}: {exc}")
            continue

        # Fetch stats if not already in DB
        season = _date_to_season(game_date)
        home_id = game.get("home_team_id", "")
        away_id = game.get("away_team_id", "")

        for team_id in [home_id, away_id]:
            if not db.get_team_stats(team_id, season):
                try:
                    stats = stats_adapter.get_team_stats(team_id, season)
                    if stats:
                        stats["team_id"] = team_id
                        stats["season"] = season
                        db.upsert_team_stats(stats)
                except Exception as exc:
                    pass  # Non-fatal

        # Build features
        if not db.get_game_features(game_id):
            try:
                home_stats = db.get_team_stats(home_id, season) or {}
                away_stats = db.get_team_stats(away_id, season) or {}
                recent_home = db.get_recent_games(home_id, 10, game_date)
                recent_away = db.get_recent_games(away_id, 10, game_date)

                features = feature_engineer.build_game_features(
                    game, home_stats, away_stats, recent_home, recent_away
                )
                db.save_game_features(game_id, features)
            except Exception as exc:
                pass  # Non-fatal

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(games)} games...")

    print(f"\nCollection complete. {len(games)} games saved.")


def cmd_dashboard(args) -> None:
    """Launch the Streamlit dashboard."""
    import subprocess
    dashboard_path = PROJECT_ROOT / "dashboard" / "app.py"
    print(f"Launching Streamlit dashboard: {dashboard_path}")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(dashboard_path)],
        check=True,
    )


def cmd_schedule_daemon(args) -> None:
    """Start the APScheduler daemon."""
    print(BANNER)
    print("Starting scheduler daemon...")
    from scheduler import start_scheduler
    start_scheduler()


# ── Argument parser ────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cbb-totals",
        description="CBB Totals Model — NCAA Men's Basketball Over/Under Projections",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # run
    p_run = sub.add_parser("run", help="Run the full daily pipeline")
    p_run.add_argument("--date", metavar="YYYY-MM-DD", help="Target date (default: today)")
    p_run.add_argument("--force", action="store_true", help="Force re-fetch stats")

    # refresh-odds
    p_refresh = sub.add_parser("refresh-odds", help="Refresh odds only (intraday)")
    p_refresh.add_argument("--date", metavar="YYYY-MM-DD", help="Target date (default: today)")

    # train
    sub.add_parser("train", help="Train / retrain ML models")

    # backtest
    p_bt = sub.add_parser("backtest", help="Run historical backtest")
    p_bt.add_argument("--start", metavar="YYYY-MM-DD", help="Start date")
    p_bt.add_argument("--end", metavar="YYYY-MM-DD", help="End date")

    # collect
    p_collect = sub.add_parser("collect", help="Collect historical game data")
    p_collect.add_argument("--start", metavar="YYYY-MM-DD", required=True, help="Start date")
    p_collect.add_argument("--end", metavar="YYYY-MM-DD", required=True, help="End date")

    # dashboard
    sub.add_parser("dashboard", help="Launch Streamlit dashboard")

    # schedule
    sub.add_parser("schedule", help="Start the scheduler daemon")

    return parser


# ── Entry point ───────────────────────────────────────────────────────────────

COMMAND_MAP = {
    "run": cmd_run,
    "refresh-odds": cmd_refresh_odds,
    "train": cmd_train,
    "backtest": cmd_backtest,
    "collect": cmd_collect,
    "dashboard": cmd_dashboard,
    "schedule": cmd_schedule_daemon,
}


def main():
    parser = build_parser()
    args = parser.parse_args()

    handler = COMMAND_MAP.get(args.command)
    if handler:
        try:
            handler(args)
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
            sys.exit(0)
        except Exception as exc:
            print(f"\nERROR: {exc}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

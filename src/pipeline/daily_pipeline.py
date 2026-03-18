"""
Daily pipeline for CBB Totals Model.
Orchestrates schedule fetch → stats → features → projection → odds → edges → output.
"""

from __future__ import annotations

import time
import traceback
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DailyPipeline:
    """
    Full daily pipeline: schedule → stats → ML → odds → edges → output.

    Usage::

        config = load_config()
        db = DatabaseManager(config)
        db.init_db()
        pipeline = DailyPipeline(config, db)
        pipeline.run()
    """

    def __init__(self, config: dict, db_manager):
        self.config = config
        self.db = db_manager

        # Lazy-imported components (avoid circular imports)
        self._schedule_adapter = None
        self._stats_adapter = None
        self._odds_adapter = None
        self._feature_engineer = None
        self._baseline_model = None
        self._ml_predictor = None
        self._ensemble_model = None
        self._edge_calculator = None
        self._alert_manager = None
        self._normalizer = None

        self.model_version = "1.0"

    # ── Component initialization ──────────────────────────────────────────

    def _init_components(self) -> None:
        """Lazy-initialize all components."""
        from src.data.base_adapter import DataAdapterFactory
        from src.data.team_normalizer import TeamNormalizer
        from src.features.feature_engineering import FeatureEngineer
        from src.models.baseline_model import BaselineModel
        from src.models.ensemble import EnsembleModel
        from src.pipeline.edge_calculator import EdgeCalculator
        from src.utils.alerts import AlertManager

        factory = DataAdapterFactory(self.config)
        self._schedule_adapter = factory.get_schedule_adapter()
        self._stats_adapter = factory.get_stats_adapter()
        self._odds_adapter = factory.get_odds_adapter()
        self._odds_adapter.db = self.db

        self._normalizer = TeamNormalizer(self.config)
        self._feature_engineer = FeatureEngineer(self.config)
        self._baseline_model = BaselineModel(self.config)
        self._ensemble_model = EnsembleModel(self.config)
        self._edge_calculator = EdgeCalculator(self.config)
        self._alert_manager = AlertManager(self.config)

        # Try to load ML models from disk
        self._load_ml_models()

    def _load_ml_models(self) -> None:
        """Load trained ML models if available."""
        from src.models.ml_model import MLModelTrainer, MLModelPredictor

        models_dir = Path(__file__).resolve().parents[2] / "models"
        model_file = models_dir / "cbb_totals_models.joblib"

        if model_file.exists():
            try:
                trainer = MLModelTrainer(self.config)
                trainer.load_models(str(models_dir))
                self._ml_predictor = MLModelPredictor(trainer)
                logger.info("ML models loaded successfully.")
            except Exception as exc:
                logger.warning(f"Could not load ML models: {exc}. Will use baseline only.")
                self._ml_predictor = None
        else:
            logger.info("No trained ML models found. Running baseline-only mode.")
            self._ml_predictor = None

    # ── Main run ──────────────────────────────────────────────────────────

    def run(
        self,
        run_date: Optional[str] = None,
        force_refresh: bool = False,
    ) -> dict:
        """
        Run the full daily pipeline.

        Args:
            run_date:      Date to run for (YYYY-MM-DD). Defaults to today.
            force_refresh: If True, re-fetch stats even if already in DB.

        Returns:
            Summary dict with keys:
            games_processed, games_with_odds, top_edges, csv_path, run_time_s.
        """
        start_time = time.time()
        target_date = run_date or date.today().strftime("%Y-%m-%d")

        logger.info(f"{'='*60}")
        logger.info(f"CBB Totals Pipeline — Running for {target_date}")
        logger.info(f"{'='*60}")

        self._init_components()

        # Determine current season
        season = _date_to_season(target_date)

        # ── Step 1: Fetch schedule ─────────────────────────────────────────
        logger.info("Step 1/7: Fetching schedule...")
        games = self._fetch_schedule(target_date)

        if not games:
            logger.warning(f"No games found for {target_date}. Pipeline complete.")
            return {
                "games_processed": 0,
                "games_with_odds": 0,
                "top_edges": [],
                "csv_path": None,
                "run_time_s": round(time.time() - start_time, 2),
            }

        logger.info(f"Found {len(games)} games for {target_date}.")

        # Save games to DB
        for game in games:
            try:
                self.db.upsert_game(game)
            except Exception as exc:
                logger.debug(f"Could not upsert game {game.get('game_id')}: {exc}")

        # ── Step 2–3: Stats + Features per game ───────────────────────────
        logger.info("Step 2/7: Fetching stats and building features...")
        game_features_map: dict[str, dict] = {}
        games_with_features = []

        for game in games:
            game_id = game.get("game_id", "")
            home_id = game.get("home_team_id", "")
            away_id = game.get("away_team_id", "")

            # Skip if already have features and not forcing refresh
            if not force_refresh:
                existing = self.db.get_game_features(game_id)
                if existing:
                    game_features_map[game_id] = existing
                    games_with_features.append(game)
                    logger.debug(f"Using cached features for {game_id}")
                    continue

            try:
                home_stats, away_stats, recent_home, recent_away = \
                    self._fetch_game_data(home_id, away_id, target_date, season)
            except Exception as exc:
                logger.warning(
                    f"Could not fetch stats for {game.get('home_team', home_id)} "
                    f"vs {game.get('away_team', away_id)}: {exc}"
                )
                continue

            # Add rest days to game dict
            game = self._add_rest_days(game, home_id, away_id, target_date)

            # Build features
            try:
                features = self._feature_engineer.build_game_features(
                    game, home_stats, away_stats, recent_home, recent_away
                )
                features["game_id"] = game_id
                features["home_team"] = game.get("home_team", "")
                features["away_team"] = game.get("away_team", "")
                features["game_date"] = target_date

                self.db.save_game_features(game_id, features)
                game_features_map[game_id] = features
                games_with_features.append(game)

            except Exception as exc:
                logger.warning(f"Feature engineering failed for {game_id}: {exc}")
                logger.debug(traceback.format_exc())

        logger.info(
            f"Features built for {len(games_with_features)}/{len(games)} games."
        )

        # ── Step 4-5: Projection per game ─────────────────────────────────
        logger.info("Step 3/7: Generating projections...")
        projections: dict[str, dict] = {}

        for game in games_with_features:
            game_id = game.get("game_id", "")
            features = game_features_map.get(game_id)
            if not features:
                continue

            try:
                projection = self._project_game(game, features)
                projections[game_id] = projection

                # Save to DB
                self.db.save_projection(
                    {
                        "game_id": game_id,
                        "baseline_total": projection.get("baseline_total"),
                        "ml_total": projection.get("ml_total"),
                        "ensemble_total": projection.get("ensemble_total"),
                        "predicted_home_score": projection.get("predicted_home_score"),
                        "predicted_away_score": projection.get("predicted_away_score"),
                        "predicted_possessions": projection.get("predicted_possessions"),
                        "confidence_score": projection.get("confidence_score"),
                        "model_version": self.model_version,
                    }
                )
            except Exception as exc:
                logger.warning(f"Projection failed for {game_id}: {exc}")
                logger.debug(traceback.format_exc())

        logger.info(f"Projections generated for {len(projections)} games.")

        # ── Step 6: Odds ──────────────────────────────────────────────────
        logger.info("Step 4/7: Fetching odds...")
        odds_by_game = self._fetch_and_match_odds(target_date, games)

        # ── Step 7: Edge computation ──────────────────────────────────────
        logger.info("Step 5/7: Computing edges...")
        all_edges = []

        for game in games_with_features:
            game_id = game.get("game_id", "")
            projection = projections.get(game_id)
            if not projection:
                continue

            # Augment with game metadata
            proj_with_meta = {
                **projection,
                "game_id": game_id,
                "game_date": target_date,
                "home_team": game.get("home_team", ""),
                "away_team": game.get("away_team", ""),
                "home_team_id": game.get("home_team_id", ""),
                "away_team_id": game.get("away_team_id", ""),
            }

            odds = odds_by_game.get(game_id)
            if odds:
                market_total = odds.get("total")
                if market_total:
                    edge = self._edge_calculator.compute_edge(proj_with_meta, market_total)
                    edge["over_price"] = odds.get("over_price")
                    edge["under_price"] = odds.get("under_price")
                    edge["sportsbook"] = odds.get("sportsbook", "consensus")

                    # Recompute confidence with actual edge magnitude
                    features = game_features_map.get(game_id, {})
                    baseline_result = {"baseline_total": projection.get("baseline_total")}
                    ml_result = {"ml_ensemble_total": projection.get("ml_total"),
                                 "model_agreement_score": 0.7}
                    conf = self._ensemble_model.compute_confidence_with_edge(
                        features, baseline_result, ml_result,
                        edge.get("abs_differential", 0)
                    )
                    edge["confidence_score"] = conf
                    edge["confidence_label"] = self._ensemble_model.interpret_confidence(conf)

                    # Save odds snapshot
                    try:
                        self.db.save_odds_snapshot(
                            {
                                "game_id": game_id,
                                "sportsbook": odds.get("sportsbook", "unknown"),
                                "market_total": market_total,
                                "over_odds": odds.get("over_price"),
                                "under_odds": odds.get("under_price"),
                            }
                        )
                    except Exception as exc:
                        logger.debug(f"Could not save odds snapshot: {exc}")

                    all_edges.append(edge)
            else:
                # No odds: still include the projection without edge
                proj_with_meta["market_total"] = None
                proj_with_meta["differential"] = None
                proj_with_meta["abs_differential"] = None
                proj_with_meta["edge_side"] = None
                proj_with_meta["edge_bucket"] = None
                proj_with_meta["formatted_differential"] = "—"
                all_edges.append(proj_with_meta)

        games_with_odds = sum(1 for e in all_edges if e.get("market_total") is not None)
        logger.info(f"Odds matched for {games_with_odds}/{len(all_edges)} games.")

        # ── Rank and output ───────────────────────────────────────────────
        logger.info("Step 6/7: Ranking edges and generating output...")
        ranked_df = self._edge_calculator.rank_edges(all_edges, min_edge=0.0)

        # CSV export
        csv_path = None
        if not ranked_df.empty:
            csv_path = self._edge_calculator.export_csv(ranked_df)

        # Console output
        if not ranked_df.empty:
            header = (
                f"\nCBB Totals — {target_date}  "
                f"({len(ranked_df)} games, {games_with_odds} with odds)\n"
            )
            print(header)
            output_str = self._edge_calculator.format_console_output(ranked_df)
            print(output_str)
        else:
            logger.info("No edges to display.")

        # Alerts
        logger.info("Step 7/7: Sending alerts...")
        if not ranked_df.empty:
            alert_threshold = float(
                self.config.get("alerts", {}).get("threshold", 6.0)
            )
            n_alerts = self._alert_manager.check_and_alert(ranked_df, alert_threshold)
            logger.info(f"Sent {n_alerts} alert(s).")

            # Daily summary
            self._alert_manager.send_daily_summary(ranked_df)

        run_time = round(time.time() - start_time, 2)
        logger.info(f"Pipeline complete in {run_time}s.")

        # Top edges for return
        top_edges = []
        if not ranked_df.empty:
            cols = ["home_team", "away_team", "market_total", "ensemble_total",
                    "differential", "edge_side", "confidence_score"]
            available = [c for c in cols if c in ranked_df.columns]
            top_edges = ranked_df[available].head(5).to_dict("records")

        return {
            "games_processed": len(games),
            "games_with_features": len(games_with_features),
            "games_with_odds": games_with_odds,
            "top_edges": top_edges,
            "csv_path": csv_path,
            "run_time_s": run_time,
        }

    # ── Intraday odds refresh ─────────────────────────────────────────────

    def refresh_odds_only(self, run_date: Optional[str] = None) -> dict:
        """
        Re-pull current odds and recompute edges against existing projections.

        Skips stats/features/projection steps.

        Args:
            run_date: Date to refresh (YYYY-MM-DD). Defaults to today.

        Returns:
            Summary dict.
        """
        start_time = time.time()
        target_date = run_date or date.today().strftime("%Y-%m-%d")
        logger.info(f"Intraday odds refresh for {target_date}...")

        self._init_components()

        # Get existing projections from DB
        existing_projections = self.db.get_todays_projections(target_date)
        if not existing_projections:
            logger.warning(f"No existing projections found for {target_date}.")
            return {"updated": 0, "run_time_s": round(time.time() - start_time, 2)}

        games = self.db.get_todays_games(target_date)
        # Refresh odds
        odds_by_game = self._fetch_and_match_odds(target_date, games)

        all_edges = []
        for proj in existing_projections:
            game_id = proj.get("game_id", "")
            odds = odds_by_game.get(game_id)
            if not odds or not odds.get("total"):
                continue

            proj_with_meta = {**proj}
            proj_with_meta["home_team"] = proj.get("home_team_id", "")
            proj_with_meta["away_team"] = proj.get("away_team_id", "")

            edge = self._edge_calculator.compute_edge(proj_with_meta, odds["total"])
            edge["sportsbook"] = odds.get("sportsbook", "consensus")
            all_edges.append(edge)

            # Save new snapshot
            try:
                self.db.save_odds_snapshot(
                    {
                        "game_id": game_id,
                        "sportsbook": odds.get("sportsbook", "unknown"),
                        "market_total": odds.get("total"),
                        "over_odds": odds.get("over_price"),
                        "under_odds": odds.get("under_price"),
                    }
                )
            except Exception as exc:
                logger.debug(f"Odds snapshot save error: {exc}")

        ranked_df = self._edge_calculator.rank_edges(all_edges, min_edge=0.0)

        if not ranked_df.empty:
            print(f"\nIntraday Refresh — {target_date}")
            print(self._edge_calculator.format_console_output(ranked_df))
            self._edge_calculator.export_csv(ranked_df)
            self._alert_manager.check_and_alert(ranked_df)

        run_time = round(time.time() - start_time, 2)
        logger.info(f"Odds refresh complete in {run_time}s. {len(all_edges)} games updated.")
        return {"updated": len(all_edges), "run_time_s": run_time}

    # ── Private helpers ───────────────────────────────────────────────────

    def _fetch_schedule(self, target_date: str) -> list[dict]:
        """Fetch schedule with up to 3 retries."""
        for attempt in range(3):
            try:
                games = self._schedule_adapter.get_schedule_by_date(target_date)
                return games
            except Exception as exc:
                logger.warning(f"Schedule fetch attempt {attempt+1}/3 failed: {exc}")
                if attempt < 2:
                    time.sleep(2 ** attempt)
        return []

    def _fetch_game_data(
        self,
        home_id: str,
        away_id: str,
        target_date: str,
        season: str,
    ) -> tuple[dict, dict, list, list]:
        """Fetch stats and recent games for both teams."""
        # Season stats
        home_stats = self.db.get_team_stats(home_id, season) or {}
        away_stats = self.db.get_team_stats(away_id, season) or {}

        # If not in DB, fetch from adapter
        if not home_stats:
            for attempt in range(3):
                try:
                    raw = self._stats_adapter.get_team_stats(home_id, season)
                    if raw:
                        home_stats = raw
                        raw["team_id"] = home_id
                        raw["season"] = season
                        self.db.upsert_team_stats(raw)
                    break
                except Exception as exc:
                    logger.debug(f"Home stats fetch attempt {attempt+1}: {exc}")
                    if attempt < 2:
                        time.sleep(1)

        if not away_stats:
            for attempt in range(3):
                try:
                    raw = self._stats_adapter.get_team_stats(away_id, season)
                    if raw:
                        away_stats = raw
                        raw["team_id"] = away_id
                        raw["season"] = season
                        self.db.upsert_team_stats(raw)
                    break
                except Exception as exc:
                    logger.debug(f"Away stats fetch attempt {attempt+1}: {exc}")
                    if attempt < 2:
                        time.sleep(1)

        # Recent games
        recent_home = self.db.get_recent_games(home_id, 10, target_date)
        recent_away = self.db.get_recent_games(away_id, 10, target_date)

        # If none in DB, try game log from adapter
        if not recent_home:
            try:
                log = self._stats_adapter.get_game_log(home_id, season)
                recent_home = [g for g in log if g.get("date", "") < target_date][-10:]
            except Exception:
                pass

        if not recent_away:
            try:
                log = self._stats_adapter.get_game_log(away_id, season)
                recent_away = [g for g in log if g.get("date", "") < target_date][-10:]
            except Exception:
                pass

        return home_stats, away_stats, recent_home, recent_away

    def _add_rest_days(
        self,
        game: dict,
        home_id: str,
        away_id: str,
        target_date: str,
    ) -> dict:
        """Add days_rest_home and days_rest_away to game dict."""
        game = game.copy()

        from src.features.recent_form import RecentFormCalculator
        calc = RecentFormCalculator()

        game["days_rest_home"] = calc.get_rest_days(home_id, target_date, self.db)
        game["days_rest_away"] = calc.get_rest_days(away_id, target_date, self.db)
        return game

    def _project_game(self, game: dict, features: dict) -> dict:
        """Run baseline + ML + ensemble projection for one game."""
        # Baseline
        baseline_result = self._baseline_model.predict(features)

        # ML (if available)
        ml_result = {}
        if self._ml_predictor and self._ml_predictor.is_trained():
            try:
                ml_result = self._ml_predictor.predict(features)
            except Exception as exc:
                logger.debug(f"ML prediction failed: {exc}")

        # Ensemble
        ensemble_result = self._ensemble_model.predict(
            baseline_result, ml_result, self.config
        )

        # Confidence
        confidence = self._ensemble_model.compute_confidence_score(
            features, baseline_result, ml_result
        )

        return {
            **ensemble_result,
            "baseline_total": baseline_result.get("baseline_total"),
            "ml_total": ml_result.get("ml_ensemble_total"),
            "confidence_score": confidence,
        }

    def _fetch_and_match_odds(
        self,
        target_date: str,
        games: list[dict],
    ) -> dict[str, dict]:
        """
        Fetch odds and match them to ESPN game IDs.

        Returns dict: game_id → best matching odds dict.
        """
        try:
            # Try consensus first, then all books
            consensus_odds = self._odds_adapter.get_all_books_consensus(target_date)
            if not consensus_odds:
                consensus_odds = self._odds_adapter.get_current_odds(target_date)
        except Exception as exc:
            logger.warning(f"Odds fetch failed: {exc}")
            return {}

        if not consensus_odds:
            logger.info("No odds available for matching.")
            return {}

        odds_by_game: dict[str, dict] = {}

        for game in games:
            game_id = game.get("game_id", "")
            espn_home = game.get("home_team", "")
            espn_away = game.get("away_team", "")

            best_match = self._normalizer.find_best_odds_match(
                espn_home, espn_away, consensus_odds
            )
            if best_match:
                odds_by_game[game_id] = best_match
            else:
                logger.debug(
                    f"No odds match: {espn_away} @ {espn_home}"
                )

        return odds_by_game


# ── Utilities ─────────────────────────────────────────────────────────────────

def _date_to_season(date_str: str) -> str:
    """
    Map a date string to the NCAA season year.
    NCAA season runs roughly Nov–Apr.
    Dates Nov–Dec belong to the upcoming season (e.g., Nov 2024 → '2025').
    Dates Jan–Apr belong to the current season (e.g., Feb 2025 → '2025').
    """
    try:
        d = date.fromisoformat(date_str)
        if d.month >= 11:
            return str(d.year + 1)
        return str(d.year)
    except ValueError:
        return str(date.today().year)

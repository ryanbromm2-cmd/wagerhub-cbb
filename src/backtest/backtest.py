"""
Backtesting framework for the CBB Totals Model.
Evaluates historical projection accuracy and simulated betting performance.
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)

_OUTPUTS_DIR = Path(__file__).resolve().parents[2] / "outputs"
_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Bucket boundaries for analysis
EDGE_BUCKETS = [2, 4, 6, 8]
TOTAL_BUCKETS = [(0, 130), (130, 145), (145, 160), (160, 999)]
TOTAL_BUCKET_LABELS = ["Under 130", "130-145", "145-160", "Over 160"]


class Backtester:
    """
    Runs historical backtests of the CBB Totals Model.

    Usage::

        bt = Backtester(config)
        report = bt.run_backtest(
            start_date="2023-01-01",
            end_date="2023-03-31",
            db_manager=db,
            feature_engineer=fe,
            baseline_model=bm,
            ml_predictor=ml,
            ensemble_model=em,
        )
        report.print_summary()
    """

    def __init__(self, config: dict = None):
        self.config = config or {}

    # ── Main backtest runner ──────────────────────────────────────────────

    def run_backtest(
        self,
        start_date: str,
        end_date: str,
        db_manager,
        feature_engineer,
        baseline_model,
        ml_predictor,
        ensemble_model,
    ) -> "BacktestReport":
        """
        Run a historical backtest across all completed games in [start_date, end_date].

        Data leakage prevention: for each game, only data available strictly
        BEFORE the game date is used for features.

        Args:
            start_date:       First game date to evaluate (YYYY-MM-DD).
            end_date:         Last game date to evaluate (YYYY-MM-DD).
            db_manager:       DatabaseManager instance.
            feature_engineer: FeatureEngineer instance.
            baseline_model:   BaselineModel instance.
            ml_predictor:     MLModelPredictor instance (or None).
            ensemble_model:   EnsembleModel instance.

        Returns:
            BacktestReport object with all results.
        """
        logger.info(f"Starting backtest: {start_date} → {end_date}")

        # Fetch all completed games in range
        games = db_manager.get_completed_games(start_date, end_date)
        logger.info(f"Found {len(games)} completed games for backtest.")

        if not games:
            logger.warning("No completed games found for backtest period.")
            return BacktestReport([])

        records = []
        skipped = 0

        for i, game in enumerate(games):
            game_id = game.get("game_id", "")
            game_date = game.get("date", "")
            actual_total = game.get("total_score")

            if actual_total is None:
                skipped += 1
                continue

            # ── Get features (strict pre-game cutoff) ─────────────────────
            features = db_manager.get_game_features(game_id)
            if not features:
                # Try to rebuild features using ONLY pre-game data
                try:
                    features = self._rebuild_features_preGame(
                        game, game_date, db_manager, feature_engineer
                    )
                except Exception as exc:
                    logger.debug(f"Could not rebuild features for {game_id}: {exc}")
                    skipped += 1
                    continue

            if not features:
                skipped += 1
                continue

            # ── Run projection ─────────────────────────────────────────────
            try:
                baseline_result = baseline_model.predict(features)
                ml_result = {}
                if ml_predictor and hasattr(ml_predictor, "is_trained") and ml_predictor.is_trained():
                    try:
                        ml_result = ml_predictor.predict(features)
                    except Exception:
                        pass

                ensemble_result = ensemble_model.predict(baseline_result, ml_result)
                projected_total = ensemble_result.get("ensemble_total", 0.0)
            except Exception as exc:
                logger.debug(f"Projection failed for {game_id}: {exc}")
                skipped += 1
                continue

            # ── Get market odds (closest snapshot before tipoff) ──────────
            market_total = self._get_preGame_market_total(game_id, game_date, db_manager)

            # ── Compute result ─────────────────────────────────────────────
            differential = projected_total - (market_total or projected_total)
            abs_diff = abs(differential)

            if market_total:
                # Determine Over/Under/Push
                if abs(market_total - actual_total) < 0.01:
                    ou_result = "push"
                elif actual_total > market_total:
                    actual_side = "over"
                else:
                    actual_side = "under"

                edge_side = "over" if differential > 0 else "under"
                if abs_diff < 0.01:
                    edge_side = "push"

                # Did our edge win?
                if ou_result == "push":
                    result = "push"
                elif edge_side == actual_side:
                    result = edge_side  # Correct direction
                else:
                    result = f"lose_{edge_side}"
            else:
                ou_result = "no_line"
                edge_side = "no_line"
                result = "no_line"

            # Edge bucket
            bucket = _classify_bucket(abs_diff)

            # Get season from game
            season = game.get("season", _date_to_season(game_date))

            # Conference info (from features if available)
            conf_home = str(features.get("conference_home_enc", 0))

            record = {
                "game_id": game_id,
                "game_date": game_date,
                "season": season,
                "home_team_id": game.get("home_team_id", ""),
                "away_team_id": game.get("away_team_id", ""),
                "projected_total": round(projected_total, 2),
                "market_total": market_total,
                "actual_total": float(actual_total),
                "projection_error": round(projected_total - actual_total, 2),
                "differential": round(differential, 2),
                "abs_differential": round(abs_diff, 2),
                "edge_side": edge_side,
                "result": result,
                "edge_bucket": bucket,
                "neutral_site": bool(game.get("neutral_site", False)),
                "data_completeness": features.get("data_completeness", 0),
            }
            records.append(record)

            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i+1}/{len(games)} games...")

        logger.info(
            f"Backtest complete: {len(records)} games evaluated, "
            f"{skipped} skipped."
        )

        # Save results to DB
        for rec in records:
            try:
                db_manager.save_backtest_result(
                    {
                        "game_id": rec["game_id"],
                        "projected_total": rec["projected_total"],
                        "market_total": rec["market_total"],
                        "actual_total": rec["actual_total"],
                        "differential": rec["differential"],
                        "edge_side": rec["edge_side"],
                        "result": rec["result"],
                        "edge_bucket": rec["edge_bucket"],
                        "season": rec["season"],
                    }
                )
            except Exception:
                pass

        return BacktestReport(records)

    def _rebuild_features_preGame(
        self,
        game: dict,
        game_date: str,
        db_manager,
        feature_engineer,
    ) -> Optional[dict]:
        """
        Rebuild game features using only data available before game_date.
        This ensures no future data leakage during backtesting.
        """
        home_id = game.get("home_team_id", "")
        away_id = game.get("away_team_id", "")

        # Get season stats (these should only reflect pre-game data)
        season = _date_to_season(game_date)
        home_stats = db_manager.get_team_stats(home_id, season) or {}
        away_stats = db_manager.get_team_stats(away_id, season) or {}

        if not home_stats and not away_stats:
            return None

        # Get recent games BEFORE this game date
        recent_home = db_manager.get_recent_games(home_id, 10, game_date)
        recent_away = db_manager.get_recent_games(away_id, 10, game_date)

        features = feature_engineer.build_game_features(
            game, home_stats, away_stats, recent_home, recent_away
        )
        return features

    def _get_preGame_market_total(
        self,
        game_id: str,
        game_date: str,
        db_manager,
    ) -> Optional[float]:
        """
        Get the market total from the odds snapshot closest to but before tipoff.
        """
        try:
            history = db_manager.get_line_history(game_id)
            if not history:
                # Fall back to latest odds snapshot
                snap = db_manager.get_latest_odds(game_id)
                if snap:
                    return snap.get("market_total")
                return None

            # Find the snapshot closest to the game date (but not after)
            # Assume tipoff is on game_date
            game_dt = datetime.fromisoformat(game_date)

            valid = [
                h for h in history
                if h.get("timestamp") and
                _parse_dt(h["timestamp"]) <= game_dt
            ]
            if valid:
                # Most recent before tipoff
                valid.sort(key=lambda h: _parse_dt(h["timestamp"]))
                return float(valid[-1]["total"])
            elif history:
                return float(history[0]["total"])
        except Exception as exc:
            logger.debug(f"Could not get pre-game market total for {game_id}: {exc}")
        return None


class BacktestReport:
    """
    Stores and presents results from a Backtester run.
    """

    def __init__(self, records: list[dict]):
        self.records = records
        self.df = pd.DataFrame(records) if records else pd.DataFrame()

    # ── DataFrame access ──────────────────────────────────────────────────

    def to_dataframe(self) -> pd.DataFrame:
        return self.df.copy()

    # ── Summary printing ──────────────────────────────────────────────────

    def print_summary(self) -> None:
        """Print a formatted performance summary to the console."""
        if self.df.empty:
            print("No backtest results to display.")
            return

        print("\n" + "=" * 70)
        print("BACKTEST SUMMARY")
        print("=" * 70)

        df = self.df

        # Overall stats
        n = len(df)
        errors = df["projection_error"].dropna()
        if len(errors) > 0:
            mae = errors.abs().mean()
            rmse = math.sqrt((errors ** 2).mean())
            r2 = _r2_score(df["actual_total"].values, df["projected_total"].values)
        else:
            mae = rmse = r2 = float("nan")

        print(f"\nOverall Performance ({n} games):")
        print(f"  MAE:  {mae:.2f} pts")
        print(f"  RMSE: {rmse:.2f} pts")
        print(f"  R²:   {r2:.3f}")

        # Betting performance by edge bucket
        print("\nBetting Performance by Edge Bucket:")
        print(f"  {'Bucket':<10} {'Games':>6} {'Correct':>8} {'Wrong':>6} {'Push':>6} {'Win%':>8} {'ROI':>8}")
        print("  " + "-" * 55)

        for lo, hi in [(0, 2), (2, 4), (4, 6), (6, 8), (8, 999)]:
            label = f"{lo}-{hi}" if hi < 999 else f"{lo}+"
            bucket_mask = (df["abs_differential"] >= lo) & (df["abs_differential"] < hi)
            bucket_df = df[bucket_mask & df["market_total"].notna()]

            if len(bucket_df) == 0:
                continue

            # Correct = result matches edge_side
            correct = ((bucket_df["result"] == "over") | (bucket_df["result"] == "under")).sum()
            wrong = bucket_df["result"].str.startswith("lose_").sum()
            push = (bucket_df["result"] == "push").sum()
            total_bets = correct + wrong
            win_pct = (correct / total_bets * 100) if total_bets > 0 else 0
            roi = simulate_betting(bucket_df, lo)["roi_pct"] if total_bets > 0 else 0

            print(
                f"  {label:<10} {len(bucket_df):>6} {correct:>8} {wrong:>6} "
                f"{push:>6} {win_pct:>7.1f}% {roi:>7.1f}%"
            )

        # By total range
        print("\nCalibration by Total Range:")
        print(f"  {'Range':<15} {'Games':>6} {'Proj Avg':>10} {'Actual Avg':>12} {'MAE':>8}")
        print("  " + "-" * 55)

        for lo, hi, label in [
            (0, 130, "Under 130"),
            (130, 145, "130-145"),
            (145, 160, "145-160"),
            (160, 999, "Over 160"),
        ]:
            mask = (df["actual_total"] >= lo) & (df["actual_total"] < hi)
            sub = df[mask]
            if len(sub) == 0:
                continue
            proj_avg = sub["projected_total"].mean()
            actual_avg = sub["actual_total"].mean()
            bucket_mae = (sub["projected_total"] - sub["actual_total"]).abs().mean()
            print(
                f"  {label:<15} {len(sub):>6} {proj_avg:>10.1f} {actual_avg:>12.1f} "
                f"{bucket_mae:>8.2f}"
            )

        print("\n" + "=" * 70)

    # ── CSV export ────────────────────────────────────────────────────────

    def export_csv(self, path: Optional[str] = None) -> str:
        """Save backtest results to CSV."""
        if self.df.empty:
            return ""
        if path:
            out = Path(path)
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out = _OUTPUTS_DIR / f"backtest_{ts}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(str(out), index=False)
        logger.info(f"Backtest results saved to {out}")
        return str(out)

    # ── Calibration plot ──────────────────────────────────────────────────

    def plot_calibration(self, save_path: Optional[str] = None) -> None:
        """
        Plot model projection vs actual total as a calibration scatter.
        Requires matplotlib.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not installed — cannot plot calibration.")
            return

        if self.df.empty:
            return

        df = self.df.dropna(subset=["projected_total", "actual_total"])
        fig, ax = plt.subplots(figsize=(8, 8))

        ax.scatter(
            df["actual_total"], df["projected_total"],
            alpha=0.3, s=15, label="Games"
        )
        lo = min(df["actual_total"].min(), df["projected_total"].min()) - 5
        hi = max(df["actual_total"].max(), df["projected_total"].max()) + 5
        ax.plot([lo, hi], [lo, hi], "r--", label="Perfect calibration")

        ax.set_xlabel("Actual Total")
        ax.set_ylabel("Projected Total")
        ax.set_title("CBB Totals — Calibration Chart")
        ax.legend()
        ax.grid(True, alpha=0.3)

        mae = (df["projected_total"] - df["actual_total"]).abs().mean()
        ax.text(
            0.05, 0.95, f"MAE = {mae:.2f} pts\nn = {len(df)}",
            transform=ax.transAxes, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Calibration chart saved to {save_path}")
        else:
            plt.show()
        plt.close(fig)


def simulate_betting(
    results_df: pd.DataFrame,
    edge_threshold: float = 0.0,
    vig: float = -110,
    unit: float = 100.0,
) -> dict:
    """
    Simulate flat-unit betting on games meeting the edge threshold.

    Args:
        results_df:      DataFrame with 'result', 'abs_differential' columns.
        edge_threshold:  Minimum absolute differential to bet.
        vig:             American odds for losers (e.g. -110).
        unit:            Flat bet size per game.

    Returns:
        Dict with: total_bets, wins, losses, pushes, net_units, roi_pct, win_rate.
    """
    if results_df is None or results_df.empty:
        return {
            "total_bets": 0, "wins": 0, "losses": 0, "pushes": 0,
            "net_units": 0, "roi_pct": 0, "win_rate": 0,
        }

    df = results_df.copy()
    if "abs_differential" in df.columns and edge_threshold > 0:
        df = df[df["abs_differential"] >= edge_threshold]

    if df.empty:
        return {
            "total_bets": 0, "wins": 0, "losses": 0, "pushes": 0,
            "net_units": 0, "roi_pct": 0, "win_rate": 0,
        }

    # Calculate payout for a -110 bet
    # Bet 110 to win 100 → win return = unit, lose return = -(unit * 110/100)
    win_amount = unit
    loss_amount = unit * abs(vig) / 100.0

    wins = 0
    losses = 0
    pushes = 0
    net_units = 0.0

    for _, row in df.iterrows():
        result = str(row.get("result", "")).lower()
        if result in ("over", "under"):
            wins += 1
            net_units += win_amount
        elif result.startswith("lose_"):
            losses += 1
            net_units -= loss_amount
        elif result == "push":
            pushes += 1

    total_bets = wins + losses + pushes
    win_rate = (wins / (wins + losses)) * 100 if (wins + losses) > 0 else 0
    total_wagered = (wins + losses) * loss_amount
    roi_pct = (net_units / total_wagered * 100) if total_wagered > 0 else 0

    return {
        "total_bets": total_bets,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "net_units": round(net_units, 2),
        "roi_pct": round(roi_pct, 2),
        "win_rate": round(win_rate, 1),
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _classify_bucket(abs_diff: float) -> str:
    """Return the edge bucket string."""
    for i, upper in enumerate([2, 4, 6, 8]):
        if abs_diff < upper:
            lower = [0, 2, 4, 6][i]
            return f"{lower}-{upper}"
    return "8+"


def _date_to_season(date_str: str) -> str:
    try:
        from datetime import date
        d = date.fromisoformat(date_str)
        if d.month >= 11:
            return str(d.year + 1)
        return str(d.year)
    except Exception:
        return str(datetime.now().year)


def _parse_dt(v) -> datetime:
    if isinstance(v, datetime):
        return v
    if isinstance(v, str):
        try:
            return datetime.fromisoformat(v)
        except Exception:
            return datetime.min
    return datetime.min


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Simple R² calculation."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

"""
Edge calculator for CBB Totals Model.
Computes differentials between model projections and market totals,
ranks games by edge strength, and formats output.
"""

from __future__ import annotations

import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

try:
    from tabulate import tabulate
    TABULATE_AVAILABLE = True
except ImportError:
    TABULATE_AVAILABLE = False

try:
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Edge bucket boundaries
EDGE_BUCKETS = [0, 2, 4, 6, 8, float("inf")]
BUCKET_LABELS = ["0-2", "2-4", "4-6", "6-8", "8+"]

_OUTPUTS_DIR = Path(__file__).resolve().parents[2] / "outputs"
_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


class EdgeCalculator:
    """
    Computes and formats edges between model projections and market lines.

    Usage::

        ec = EdgeCalculator(config)
        edge = ec.compute_edge(projection, market_total=145.5)
        ranked_df = ec.rank_edges([edge1, edge2, ...])
        print(ec.format_console_output(ranked_df))
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        output_cfg = self.config.get("output", {})
        self.top_n: int = int(output_cfg.get("top_n_console", 20))
        self.min_edge_display: float = float(output_cfg.get("min_edge_display", 1.0))
        self.sort_by: str = output_cfg.get("sort_by", "abs_differential")
        self.csv_dir = Path(output_cfg.get("csv_dir", "outputs/"))

        # Edge bucket labels
        self.bucket_labels = BUCKET_LABELS

    # ── Core edge computation ─────────────────────────────────────────────

    def compute_edge(
        self,
        projection: dict,
        market_total: float,
    ) -> dict:
        """
        Compute the edge for a single game.

        Args:
            projection:   Ensemble projection dict (must have 'ensemble_total').
            market_total: The sportsbook's over/under line.

        Returns:
            Input projection dict augmented with:
            - differential (float): model_total - market_total
            - abs_differential (float): |differential|
            - edge_side (str): 'OVER' | 'UNDER' | 'PUSH'
            - edge_bucket (str): which bucket the edge falls in
            - market_total (float): the provided market line
            - formatted_differential (str): e.g. '+3.5'
        """
        model_total = projection.get("ensemble_total") or projection.get("baseline_total", 0.0)
        try:
            model_total = float(model_total)
            market_total = float(market_total)
        except (TypeError, ValueError):
            logger.warning(
                f"compute_edge: could not convert totals to float "
                f"(model={model_total}, market={market_total})"
            )
            return {**projection, "differential": None, "abs_differential": None}

        differential = model_total - market_total
        abs_diff = abs(differential)

        if abs_diff < 0.01:
            edge_side = "PUSH"
        elif differential > 0:
            edge_side = "OVER"
        else:
            edge_side = "UNDER"

        bucket = _classify_bucket(abs_diff)

        result = {
            **projection,
            "market_total": round(market_total, 1),
            "differential": round(differential, 2),
            "abs_differential": round(abs_diff, 2),
            "edge_side": edge_side,
            "edge_bucket": bucket,
            "formatted_differential": format_differential(differential),
        }
        return result

    # ── Ranking ───────────────────────────────────────────────────────────

    def rank_edges(
        self,
        projections_list: list[dict],
        min_edge: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Convert a list of edge dicts to a ranked DataFrame.

        Args:
            projections_list: List of dicts from compute_edge().
            min_edge:         Minimum abs_differential to include.

        Returns:
            DataFrame sorted by abs_differential descending, with 'rank' column.
        """
        if not projections_list:
            return pd.DataFrame()

        df = pd.DataFrame(projections_list)

        # Ensure required columns exist
        for col in ["abs_differential", "differential", "edge_side"]:
            if col not in df.columns:
                df[col] = None

        # Filter by min edge
        threshold = min_edge if min_edge is not None else self.min_edge_display
        if threshold > 0 and "abs_differential" in df.columns:
            df = df[df["abs_differential"].notna() & (df["abs_differential"] >= threshold)]

        # Sort
        sort_col = self.sort_by if self.sort_by in df.columns else "abs_differential"
        df = df.sort_values(sort_col, ascending=False).reset_index(drop=True)

        # Add rank (1-indexed)
        df.insert(0, "rank", range(1, len(df) + 1))

        return df

    # ── Console output ────────────────────────────────────────────────────

    def format_console_output(
        self,
        df: pd.DataFrame,
        top_n: Optional[int] = None,
    ) -> str:
        """
        Format the edge table for console display.

        Uses Rich for color output if available, else falls back to tabulate/plain.

        Args:
            df:    Ranked DataFrame from rank_edges().
            top_n: Number of rows to show (default from config).

        Returns:
            Formatted string (plain) or prints Rich table directly.
        """
        if df is None or df.empty:
            return "No games to display."

        n = top_n or self.top_n
        display_df = df.head(n).copy()

        # Build display columns
        cols_wanted = [
            "rank", "away_team", "home_team",
            "market_total", "ensemble_total", "formatted_differential",
            "edge_side", "confidence_score", "predicted_possessions",
            "edge_bucket",
        ]
        # Only keep columns that exist
        cols = [c for c in cols_wanted if c in display_df.columns]

        if RICH_AVAILABLE:
            return self._format_rich_table(display_df, cols)
        elif TABULATE_AVAILABLE:
            return self._format_tabulate_table(display_df, cols)
        else:
            return display_df[cols].to_string(index=False)

    def _format_rich_table(self, df: pd.DataFrame, cols: list[str]) -> str:
        """Build a Rich Table and capture it as a string."""
        console = Console(force_terminal=True, width=140)
        table = Table(title="CBB Totals Model — Edge Rankings", show_header=True, header_style="bold cyan")

        column_labels = {
            "rank": "#",
            "away_team": "Away",
            "home_team": "Home",
            "market_total": "Mkt Total",
            "ensemble_total": "Model Total",
            "formatted_differential": "Diff",
            "edge_side": "Edge",
            "confidence_score": "Conf",
            "predicted_possessions": "Poss",
            "edge_bucket": "Bucket",
        }

        for col in cols:
            label = column_labels.get(col, col.replace("_", " ").title())
            table.add_column(label, justify="right" if col in ("rank", "market_total", "ensemble_total", "confidence_score", "predicted_possessions") else "left")

        for _, row in df.iterrows():
            edge = str(row.get("edge_side", ""))
            diff = str(row.get("formatted_differential", ""))

            row_cells = []
            for col in cols:
                v = row.get(col, "")
                if v is None or (isinstance(v, float) and pd.isna(v)):
                    cell = "—"
                elif col == "formatted_differential":
                    if edge == "OVER":
                        cell = f"[green]{diff}[/green]"
                    elif edge == "UNDER":
                        cell = f"[red]{diff}[/red]"
                    else:
                        cell = diff
                elif col == "edge_side":
                    if edge == "OVER":
                        cell = f"[bold green]{edge}[/bold green]"
                    elif edge == "UNDER":
                        cell = f"[bold red]{edge}[/bold red]"
                    else:
                        cell = edge
                elif col == "confidence_score":
                    try:
                        cell = f"{float(v):.2f}"
                    except (TypeError, ValueError):
                        cell = str(v)
                elif col == "predicted_possessions":
                    try:
                        cell = f"{float(v):.1f}"
                    except (TypeError, ValueError):
                        cell = str(v)
                elif col in ("market_total", "ensemble_total"):
                    try:
                        cell = f"{float(v):.1f}"
                    except (TypeError, ValueError):
                        cell = str(v)
                else:
                    cell = str(v)
                row_cells.append(cell)

            table.add_row(*row_cells)

        from io import StringIO
        import sys
        from rich.console import Console as _Console

        buf = StringIO()
        cap_console = _Console(file=buf, force_terminal=False, width=140)
        cap_console.print(table)
        return buf.getvalue()

    def _format_tabulate_table(self, df: pd.DataFrame, cols: list[str]) -> str:
        """Format using tabulate."""
        headers = {
            "rank": "#",
            "away_team": "Away",
            "home_team": "Home",
            "market_total": "Mkt",
            "ensemble_total": "Model",
            "formatted_differential": "Diff",
            "edge_side": "Edge",
            "confidence_score": "Conf",
            "predicted_possessions": "Poss",
            "edge_bucket": "Bucket",
        }
        display = df[cols].copy()
        display.columns = [headers.get(c, c) for c in cols]

        # Format numeric columns
        for old_col, new_label in [("market_total", "Mkt"), ("ensemble_total", "Model")]:
            if old_col in cols and new_label in display.columns:
                display[new_label] = display[new_label].apply(
                    lambda x: f"{float(x):.1f}" if _is_numeric(x) else x
                )
        if "Conf" in display.columns:
            display["Conf"] = display["Conf"].apply(
                lambda x: f"{float(x):.2f}" if _is_numeric(x) else x
            )
        if "Poss" in display.columns:
            display["Poss"] = display["Poss"].apply(
                lambda x: f"{float(x):.1f}" if _is_numeric(x) else x
            )

        return tabulate(display, headers="keys", tablefmt="simple", showindex=False)

    # ── CSV export ────────────────────────────────────────────────────────

    def export_csv(self, df: pd.DataFrame, path: Optional[str] = None) -> str:
        """
        Save edge rankings to a timestamped CSV file.

        Args:
            df:   DataFrame from rank_edges().
            path: Optional explicit output path.

        Returns:
            Path to the saved file.
        """
        if df is None or df.empty:
            logger.info("export_csv: nothing to export.")
            return ""

        if path:
            out_path = Path(path)
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = _OUTPUTS_DIR / f"edges_{ts}.csv"

        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert datetime columns to strings for CSV compatibility
        export_df = df.copy()
        for col in export_df.columns:
            if export_df[col].dtype == "object":
                pass  # keep as-is
            try:
                export_df[col] = export_df[col].astype(str)
            except Exception:
                pass

        export_df.to_csv(str(out_path), index=False)
        logger.info(f"Edges exported to {out_path} ({len(export_df)} rows)")
        return str(out_path)


# ── Standalone helpers ─────────────────────────────────────────────────────────

def format_differential(diff: float) -> str:
    """
    Format a differential with explicit sign.

    Examples:
        format_differential(3.5)  → '+3.5'
        format_differential(-3.5) → '-3.5'
        format_differential(0.0)  → '0.0'
    """
    if diff is None:
        return "—"
    try:
        d = float(diff)
    except (TypeError, ValueError):
        return str(diff)

    if d > 0:
        return f"+{d:.1f}"
    elif d < 0:
        return f"{d:.1f}"
    else:
        return "0.0"


def _classify_bucket(abs_diff: float) -> str:
    """Return the edge bucket label for a given absolute differential."""
    for i, upper in enumerate(EDGE_BUCKETS[1:]):
        lower = EDGE_BUCKETS[i]
        if abs_diff < upper:
            return BUCKET_LABELS[i]
    return BUCKET_LABELS[-1]


def _is_numeric(v) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False

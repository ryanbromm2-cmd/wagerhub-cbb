"""
Adapter for Bart Torvik's T-Rank public data (barttorvik.com).
Provides adjusted efficiency ratings, tempo, and game predictions.
"""

from __future__ import annotations

import io
import time
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.data.base_adapter import BaseStatsAdapter
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Torvik endpoints ──────────────────────────────────────────────────────────
_TORVIK_BASE = "https://barttorvik.com"
# CSV download: year param is the ending year of the season (e.g. 2024 = 2023-24)
_TRANK_CSV = f"{_TORVIK_BASE}/trank.php"
# Game predictions (when available)
_GAME_PREDS = f"{_TORVIK_BASE}/cbbgames.php"

# Column mapping from Torvik CSV → internal schema
_TORVIK_COLS = {
    "Team": "team_name",
    "Conf": "conference",
    "G": "games_played",
    "Rec": "record",
    "AdjOE": "adj_oe",
    "AdjDE": "adj_de",
    "Barthag": "barthag",
    "EFG%": "efg_pct",
    "EFGD%": "opp_efg_pct",
    "TOR": "tov_rate",       # Turnover rate (TO/possession)
    "TORD": "opp_tov_rate",
    "ORB": "orb_rate",       # Offensive rebound rate
    "DRB": "drb_rate",
    "FTR": "ft_rate",        # Free throw rate (FTA/FGA)
    "FTRD": "opp_ft_rate",
    "2P%": "two_p_pct",
    "2PD%": "opp_two_p_pct",
    "3P%": "three_p_pct",
    "3PD%": "opp_three_p_pct",
    "3PR": "three_pa_rate",   # 3PA / FGA
    "3PRD": "opp_three_pa_rate",
    "AdjTempo": "adj_tempo",
    "Rk": "rank",
    "Wab": "sos",             # Wins above bubble (proxy for SOS)
}

# Raw directory for caching
_RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
_RAW_DIR.mkdir(parents=True, exist_ok=True)


class TorVikAdapter(BaseStatsAdapter):
    """
    Fetches and caches Bart Torvik T-Rank efficiency data.

    Data is cached locally in data/raw/ with date-stamped filenames.
    Stale cache (>24 h old) is refreshed automatically.
    """

    def __init__(self, config: dict):
        self.config = config
        torvik_cfg = config.get("torvik", {})
        self.enabled: bool = bool(torvik_cfg.get("enabled", True))
        self.base_url: str = torvik_cfg.get("base_url", _TORVIK_BASE)
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (compatible; CBBTotalsModel/1.0)"
                )
            }
        )
        # In-memory cache: season → DataFrame
        self._cache: dict[str, pd.DataFrame] = {}

        try:
            from src.data.team_normalizer import TeamNormalizer
            self._normalizer: Optional[TeamNormalizer] = TeamNormalizer(config)
        except Exception:
            self._normalizer = None

    # ── HTTP helper ───────────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=15),
        retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
        reraise=True,
    )
    def _fetch_text(self, url: str, params: dict = None) -> str:
        time.sleep(0.5)
        resp = self._session.get(url, params=params, timeout=20)
        resp.raise_for_status()
        return resp.text

    # ── Cache helpers ─────────────────────────────────────────────────────

    def _cache_path(self, season: str) -> Path:
        today_str = date.today().strftime("%Y%m%d")
        return _RAW_DIR / f"torvik_trank_{season}_{today_str}.csv"

    def _latest_cache_file(self, season: str) -> Optional[Path]:
        """Return the most recent cache file for a season, or None."""
        files = sorted(_RAW_DIR.glob(f"torvik_trank_{season}_*.csv"), reverse=True)
        if files:
            return files[0]
        return None

    def _is_cache_fresh(self, cache_file: Path, max_age_hours: int = 24) -> bool:
        if not cache_file.exists():
            return False
        age_seconds = time.time() - cache_file.stat().st_mtime
        return age_seconds < (max_age_hours * 3600)

    # ── Core fetch ────────────────────────────────────────────────────────

    def get_team_ratings(self, season: str) -> pd.DataFrame:
        """
        Fetch T-Rank ratings for all D1 teams in a given season.

        Args:
            season: Ending year of the season as string (e.g. '2024').

        Returns:
            DataFrame with columns matching _TORVIK_COLS values.
        """
        if not self.enabled:
            logger.debug("Torvik adapter is disabled in config.")
            return pd.DataFrame()

        if season in self._cache:
            return self._cache[season]

        # Check disk cache
        existing = self._latest_cache_file(season)
        if existing and self._is_cache_fresh(existing):
            logger.debug(f"Loading Torvik ratings from cache: {existing}")
            try:
                df = pd.read_csv(str(existing))
                self._cache[season] = df
                return df
            except Exception as exc:
                logger.warning(f"Cache read failed: {exc}. Re-fetching.")

        # Fetch from web
        logger.info(f"Fetching Torvik T-Rank data for season {season}...")
        try:
            params = {"year": season, "csv": "1"}
            text_data = self._fetch_text(_TRANK_CSV, params=params)
            df = self._parse_trank_csv(text_data, season)
            if df.empty:
                logger.warning(f"Torvik returned empty data for season {season}.")
                return df

            # Save to cache
            cache_file = self._cache_path(season)
            df.to_csv(str(cache_file), index=False)
            logger.info(
                f"Torvik: {len(df)} teams loaded for season {season}. "
                f"Cached to {cache_file.name}."
            )
            self._cache[season] = df
            return df

        except Exception as exc:
            logger.error(f"Failed to fetch Torvik data for season {season}: {exc}")
            return pd.DataFrame()

    def _parse_trank_csv(self, text_data: str, season: str) -> pd.DataFrame:
        """
        Parse Torvik CSV into a normalized DataFrame.

        Torvik CSV has a header row followed by data rows.
        Some column names may vary slightly by season.
        """
        try:
            df = pd.read_csv(io.StringIO(text_data), header=0)
        except Exception as exc:
            logger.error(f"Could not parse Torvik CSV: {exc}")
            return pd.DataFrame()

        if df.empty:
            return df

        # Strip whitespace from column names
        df.columns = [c.strip() for c in df.columns]

        # Rename columns we know about
        rename_map = {}
        for raw_col, internal_col in _TORVIK_COLS.items():
            if raw_col in df.columns:
                rename_map[raw_col] = internal_col

        df = df.rename(columns=rename_map)

        # Add season column
        df["season"] = season

        # Normalize team names if normalizer available
        if self._normalizer and "team_name" in df.columns:
            df["canonical_name"] = df["team_name"].apply(
                lambda n: self._normalizer.normalize(str(n)) if pd.notna(n) else n
            )
        else:
            df["canonical_name"] = df.get("team_name", pd.Series(dtype=str))

        # Convert numeric columns
        numeric_cols = [
            "adj_oe", "adj_de", "adj_tempo", "efg_pct", "opp_efg_pct",
            "tov_rate", "opp_tov_rate", "orb_rate", "drb_rate",
            "ft_rate", "opp_ft_rate", "two_p_pct", "opp_two_p_pct",
            "three_p_pct", "opp_three_p_pct", "three_pa_rate", "opp_three_pa_rate",
            "games_played", "rank", "sos", "barthag",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Derive ppg / opp_ppg from efficiency if available
        # PPG ≈ AdjOE * (AdjTempo / 100)
        if "adj_oe" in df.columns and "adj_tempo" in df.columns:
            df["ppg_est"] = df["adj_oe"] * (df["adj_tempo"] / 100.0)
        if "adj_de" in df.columns and "adj_tempo" in df.columns:
            df["opp_ppg_est"] = df["adj_de"] * (df["adj_tempo"] / 100.0)

        return df

    def get_team_stats_by_name(self, team_name: str, season: str) -> Optional[dict]:
        """
        Look up a team's Torvik stats by name (normalized).

        Args:
            team_name: Team name (will be normalized).
            season:    Season year string.

        Returns:
            Dict of stats or None if not found.
        """
        df = self.get_team_ratings(season)
        if df.empty:
            return None

        canonical = (
            self._normalizer.normalize(team_name)
            if self._normalizer
            else team_name
        )

        # Try canonical name first
        search_col = "canonical_name" if "canonical_name" in df.columns else "team_name"
        mask = df[search_col].str.lower() == canonical.lower()
        matches = df[mask]

        if matches.empty:
            # Fuzzy fallback
            if "team_name" in df.columns:
                from difflib import get_close_matches
                all_names = df["team_name"].tolist()
                close = get_close_matches(team_name, all_names, n=1, cutoff=0.7)
                if close:
                    matches = df[df["team_name"] == close[0]]

        if matches.empty:
            logger.debug(f"Torvik: no match found for '{team_name}' in season {season}.")
            return None

        row = matches.iloc[0]
        return row.to_dict()

    # ── BaseStatsAdapter interface ────────────────────────────────────────

    def get_team_stats(self, team_id: str, season: str) -> dict:
        """
        Get team stats by ID.

        For Torvik we don't have IDs, so we try to look up by name
        from the teams table if a DB manager is available.
        Falls back to returning the full ratings DataFrame as a list.
        """
        # team_id might be 'espn_123' or a canonical name
        # Try as name directly
        stats = self.get_team_stats_by_name(team_id, season)
        if stats:
            stats["team_id"] = team_id
            stats["season"] = season
        return stats or {}

    def get_game_log(self, team_id: str, season: str) -> list[dict]:
        """
        Torvik does not offer per-game logs via public CSV.
        Returns empty list — game logs should use ESPN adapter.
        """
        logger.debug(
            f"Torvik adapter does not provide game logs (team_id={team_id}). "
            "Use ESPN adapter for game logs."
        )
        return []

    # ── Game predictions ──────────────────────────────────────────────────

    def get_game_predictions(self, date_str: str) -> list[dict]:
        """
        Fetch Torvik's game predictions for a given date (if available).

        Torvik's cbbgames.php page returns game predictions.
        This is a best-effort parse of his HTML/JSON output.

        Args:
            date_str: Date in YYYY-MM-DD format.

        Returns:
            List of prediction dicts with: home_team, away_team,
            predicted_home_score, predicted_away_score, predicted_total.
        """
        if not self.enabled:
            return []

        espn_date = date_str.replace("-", "")
        params = {"date": espn_date, "csv": "1"}

        try:
            text_data = self._fetch_text(_GAME_PREDS, params=params)
            df = pd.read_csv(io.StringIO(text_data), header=0)
            df.columns = [c.strip() for c in df.columns]
            predictions = []
            for _, row in df.iterrows():
                pred = {
                    "home_team": str(row.get("home", row.get("Home", ""))),
                    "away_team": str(row.get("away", row.get("Away", ""))),
                    "predicted_home_score": _safe_float(
                        row.get("hpred", row.get("HPred"))
                    ),
                    "predicted_away_score": _safe_float(
                        row.get("apred", row.get("APred"))
                    ),
                }
                if pred["predicted_home_score"] and pred["predicted_away_score"]:
                    pred["predicted_total"] = (
                        pred["predicted_home_score"] + pred["predicted_away_score"]
                    )
                predictions.append(pred)
            logger.info(f"Torvik game predictions for {date_str}: {len(predictions)} games.")
            return predictions
        except Exception as exc:
            logger.debug(f"Could not fetch Torvik game predictions for {date_str}: {exc}")
            return []

    # ── Utility ───────────────────────────────────────────────────────────

    def get_all_teams(self) -> list[dict]:
        """Return all teams in the current season's T-Rank data."""
        season = str(date.today().year)
        df = self.get_team_ratings(season)
        if df.empty:
            return []
        teams = []
        for _, row in df.iterrows():
            teams.append(
                {
                    "team_name": row.get("team_name", ""),
                    "canonical_name": row.get("canonical_name", ""),
                    "conference": row.get("conference", ""),
                    "season": season,
                    "adj_oe": row.get("adj_oe"),
                    "adj_de": row.get("adj_de"),
                    "adj_tempo": row.get("adj_tempo"),
                }
            )
        return teams

    def clear_cache(self) -> None:
        """Clear in-memory cache."""
        self._cache.clear()


def _safe_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

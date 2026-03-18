"""
The Odds API v4 adapter for NCAA Men's Basketball totals.
https://the-odds-api.com
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from datetime import datetime
from typing import Optional

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.data.base_adapter import BaseOddsAdapter
from src.utils.logger import get_logger

logger = get_logger(__name__)

_BASE_URL = "https://api.the-odds-api.com/v4"
_SPORT = "basketball_ncaab"


class TheOddsAPIAdapter(BaseOddsAdapter):
    """
    Fetches NCAA basketball totals from The Odds API v4.

    Requires THE_ODDS_API_KEY environment variable (or config).
    Falls back to empty results with a warning if key is absent.

    Free tier: 500 requests/month.  Each call to get_current_odds
    consumes 1 request per bookmaker.  Use sparingly.
    """

    def __init__(self, config: dict, db_manager=None):
        self.config = config
        self.db = db_manager

        odds_cfg = config.get("the_odds_api", {})
        self.base_url: str = odds_cfg.get("base_url", _BASE_URL)
        self.sport: str = odds_cfg.get("sport", _SPORT)
        self.markets: str = odds_cfg.get("markets", "totals")
        self.regions: str = odds_cfg.get("regions", "us")
        self.odds_format: str = odds_cfg.get("odds_format", "american")

        self.api_key: str = os.environ.get("THE_ODDS_API_KEY", "")
        if not self.api_key:
            logger.warning(
                "THE_ODDS_API_KEY is not set. "
                "Odds fetching will return empty results."
            )

        self._session = requests.Session()
        self._remaining_requests: Optional[int] = None
        self._remaining_usage: Optional[int] = None

    # ── HTTP helpers ──────────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
        reraise=True,
    )
    def _get(self, url: str, params: dict = None) -> dict | list:
        """HTTP GET with retry; updates quota tracking from headers."""
        if not self.api_key:
            return []

        resp = self._session.get(url, params=params, timeout=20)

        # Track quota from response headers
        self._remaining_requests = _header_int(resp, "x-requests-remaining")
        self._remaining_usage = _header_int(resp, "x-requests-used")

        if resp.status_code == 401:
            logger.error("The Odds API: 401 Unauthorized — check your API key.")
            return []
        if resp.status_code == 429:
            logger.warning("The Odds API: 429 Too Many Requests — quota exceeded.")
            return []

        resp.raise_for_status()
        return resp.json()

    # ── Core odds fetch ───────────────────────────────────────────────────

    def get_current_odds(self, date: Optional[str] = None) -> list[dict]:
        """
        Fetch current totals for all upcoming NCAAB games.

        Args:
            date: Optional date filter (YYYY-MM-DD). The Odds API returns
                  all upcoming games; this filters by commence_time date.

        Returns:
            List of odds dicts:
            {game_id, home_team, away_team, commence_time,
             sportsbook, total, over_price, under_price}
        """
        url = f"{self.base_url}/sports/{self.sport}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": self.regions,
            "markets": self.markets,
            "oddsFormat": self.odds_format,
            "dateFormat": "iso",
        }

        logger.debug(f"Fetching odds from The Odds API (sport={self.sport})")
        try:
            raw = self._get(url, params=params)
        except Exception as exc:
            logger.error(f"The Odds API fetch failed: {exc}")
            return []

        if not isinstance(raw, list):
            logger.warning(f"Unexpected Odds API response type: {type(raw)}")
            return []

        self.check_remaining_requests()
        results = self._parse_odds_response(raw)

        # Filter by date if requested
        if date:
            results = [
                r for r in results
                if r.get("commence_time", "")[:10] == date
            ]

        # Persist to line_history if DB manager available
        if self.db and results:
            self._persist_line_history(results)

        logger.info(
            f"The Odds API: {len(results)} odds entries "
            f"({'filtered for ' + date if date else 'all upcoming'})."
        )
        return results

    def _parse_odds_response(self, raw: list) -> list[dict]:
        """
        Parse The Odds API v4 response into flat odds dicts.

        Response structure:
        [
          {
            "id": "abc123",
            "sport_key": "basketball_ncaab",
            "home_team": "Duke Blue Devils",
            "away_team": "North Carolina Tar Heels",
            "commence_time": "2024-02-03T21:00:00Z",
            "bookmakers": [
              {
                "key": "draftkings",
                "title": "DraftKings",
                "last_update": "...",
                "markets": [
                  {
                    "key": "totals",
                    "outcomes": [
                      {"name": "Over", "price": -110, "point": 145.5},
                      {"name": "Under", "price": -110, "point": 145.5}
                    ]
                  }
                ]
              }
            ]
          }
        ]
        """
        results = []

        for game in raw:
            game_id = game.get("id", "")
            home_team = game.get("home_team", "")
            away_team = game.get("away_team", "")
            commence_time = game.get("commence_time", "")

            bookmakers = game.get("bookmakers", [])
            for book in bookmakers:
                book_key = book.get("key", "")
                markets = book.get("markets", [])

                for market in markets:
                    if market.get("key") != "totals":
                        continue

                    outcomes = market.get("outcomes", [])
                    over_outcome = next(
                        (o for o in outcomes if o.get("name") == "Over"), None
                    )
                    under_outcome = next(
                        (o for o in outcomes if o.get("name") == "Under"), None
                    )

                    if not over_outcome:
                        continue

                    total_line = over_outcome.get("point")
                    over_price = over_outcome.get("price")
                    under_price = under_outcome.get("price") if under_outcome else None

                    results.append(
                        {
                            "game_id": f"odds_{game_id}",
                            "odds_api_id": game_id,
                            "home_team": home_team,
                            "away_team": away_team,
                            "commence_time": commence_time,
                            "sportsbook": book_key,
                            "total": _safe_float(total_line),
                            "over_price": _safe_int(over_price),
                            "under_price": _safe_int(under_price),
                        }
                    )

        return results

    # ── Secondary methods ─────────────────────────────────────────────────

    def get_odds_by_game(self, game_id: str) -> list[dict]:
        """
        Return all sportsbook odds for a specific game.

        Note: The Odds API doesn't support per-game queries directly.
        This filters from the full odds pull.

        Args:
            game_id: Either our 'odds_' prefixed ID or the raw Odds API ID.
        """
        all_odds = self.get_current_odds()
        raw_id = game_id.replace("odds_", "")
        return [
            o for o in all_odds
            if o.get("odds_api_id") == raw_id or o.get("game_id") == game_id
        ]

    def get_all_books_consensus(self, date: Optional[str] = None) -> list[dict]:
        """
        Return consensus (average) total per game across all available books.
        Only includes books in the configured list if provided.
        """
        all_odds = self.get_current_odds(date)
        if not all_odds:
            return []

        # Desired books from config
        cfg_books = (
            self.config.get("data_sources", {})
            .get("odds", {})
            .get("books", [])
        )

        # Group by odds_api_id
        groups: dict[str, list] = defaultdict(list)
        for entry in all_odds:
            if cfg_books and entry.get("sportsbook") not in cfg_books:
                continue
            groups[entry.get("odds_api_id", "")].append(entry)

        consensus_list = []
        for api_id, book_entries in groups.items():
            totals = [e["total"] for e in book_entries if e.get("total") is not None]
            if not totals:
                continue

            avg_total = sum(totals) / len(totals)
            # Round to nearest 0.5
            rounded = round(avg_total * 2) / 2

            rep = book_entries[0].copy()
            rep["sportsbook"] = "consensus"
            rep["total"] = rounded
            rep["books_included"] = len(totals)
            rep["all_totals"] = totals

            # Average prices
            over_prices = [e["over_price"] for e in book_entries if e.get("over_price")]
            under_prices = [e["under_price"] for e in book_entries if e.get("under_price")]
            rep["over_price"] = round(sum(over_prices) / len(over_prices)) if over_prices else -110
            rep["under_price"] = round(sum(under_prices) / len(under_prices)) if under_prices else -110

            consensus_list.append(rep)

        return consensus_list

    def get_opening_lines(self) -> list[dict]:
        """
        Return the earliest available line for each game.
        Uses The Odds API's historical endpoint if available,
        otherwise returns current lines flagged as 'opening'.

        Note: Historical odds endpoint requires paid tier.
        """
        logger.info(
            "Opening line history requires The Odds API paid tier. "
            "Returning current lines."
        )
        odds = self.get_current_odds()
        for o in odds:
            o["is_opening"] = True
        return odds

    def check_remaining_requests(self) -> None:
        """Log the remaining API request quota."""
        if self._remaining_requests is not None:
            logger.info(
                f"The Odds API quota: {self._remaining_requests} requests remaining "
                f"({self._remaining_usage} used)."
            )

    # ── Persistence helpers ────────────────────────────────────────────────

    def _persist_line_history(self, odds_list: list[dict]) -> None:
        """Save all fetched odds to the line_history table."""
        if not self.db:
            return
        now = datetime.utcnow()
        for entry in odds_list:
            try:
                self.db.save_line_history(
                    {
                        "game_id": entry.get("game_id", ""),
                        "sportsbook": entry.get("sportsbook", ""),
                        "total": entry.get("total"),
                        "timestamp": now,
                    }
                )
            except Exception as exc:
                logger.debug(f"Could not save line history entry: {exc}")

    def save_snapshots_to_db(self, odds_list: list[dict], is_opening: bool = False) -> int:
        """
        Persist odds snapshots to the DB.

        Args:
            odds_list:  List returned by get_current_odds or get_all_books_consensus.
            is_opening: Mark these entries as the opening line.

        Returns:
            Number of records saved.
        """
        if not self.db:
            logger.warning("No DB manager set; cannot save odds snapshots.")
            return 0

        saved = 0
        now = datetime.utcnow()
        for entry in odds_list:
            try:
                self.db.save_odds_snapshot(
                    {
                        "game_id": entry.get("game_id", ""),
                        "sportsbook": entry.get("sportsbook", "unknown"),
                        "market_total": entry.get("total"),
                        "over_odds": entry.get("over_price"),
                        "under_odds": entry.get("under_price"),
                        "snapshot_time": now,
                        "is_opening": is_opening,
                        "is_closing": False,
                    }
                )
                saved += 1
            except Exception as exc:
                logger.debug(f"Snapshot save error: {exc}")

        logger.debug(f"Saved {saved} odds snapshots to DB.")
        return saved


# ── Utilities ─────────────────────────────────────────────────────────────────

def _safe_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _header_int(response: requests.Response, header: str) -> Optional[int]:
    try:
        return int(response.headers.get(header, ""))
    except (TypeError, ValueError):
        return None

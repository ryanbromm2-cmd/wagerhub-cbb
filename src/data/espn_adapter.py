"""
ESPN public API adapter for schedule and team statistics.
No API key required — uses ESPN's undocumented public JSON endpoints.
"""

from __future__ import annotations

import time
from datetime import date, timedelta
from typing import Optional

import requests
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from src.data.base_adapter import BaseScheduleAdapter, BaseStatsAdapter
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── ESPN endpoints ─────────────────────────────────────────────────────────────
_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"
_SCOREBOARD = f"{_BASE}/scoreboard"
_TEAMS = f"{_BASE}/teams"
_SUMMARY = f"{_BASE}/summary"

# ESPN stat category name → our internal stat key
_STAT_MAP: dict[str, str] = {
    "avgPoints": "ppg",
    "avgPointsAllowed": "opp_ppg",
    "fieldGoalPct": "efg_pct",          # ESPN uses FG%, we treat as proxy
    "threePointFieldGoalPct": "three_p_pct",
    "freeThrowPct": "ft_pct",
    "avgRebounds": "avg_rebounds",
    "avgAssists": "avg_assists",
    "avgTurnovers": "tov_per_game",
    "avgSteals": "avg_steals",
    "avgBlocks": "avg_blocks",
}


class ESPNScheduleAdapter(BaseScheduleAdapter):
    """
    Fetches CBB game schedules via ESPN's public scoreboard API.

    The scoreboard endpoint supports a `dates` parameter (YYYYMMDD) to
    retrieve games for any past or future date.
    """

    def __init__(self, config: dict):
        self.config = config
        espn_cfg = config.get("espn", {})
        self.request_delay: float = float(espn_cfg.get("request_delay", 0.5))
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (compatible; CBBTotalsModel/1.0; "
                    "+https://github.com/cbb-totals)"
                )
            }
        )

    # ── Internal helpers ──────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
        reraise=True,
    )
    def _get(self, url: str, params: dict = None) -> dict:
        """HTTP GET with retry."""
        time.sleep(self.request_delay)
        resp = self._session.get(url, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def _parse_scoreboard(self, data: dict) -> list[dict]:
        """
        Extract game information from ESPN scoreboard JSON.

        ESPN scoreboard structure:
        {
          "events": [
            {
              "id": "...",
              "date": "2024-01-15T00:00Z",
              "name": "Duke Blue Devils at North Carolina Tar Heels",
              "competitions": [{
                "id": "...",
                "neutralSite": false,
                "status": {"type": {"name": "STATUS_FINAL"}},
                "competitors": [
                  {"homeAway": "home", "team": {"id": "...", "displayName": "..."}, "score": "78"},
                  {"homeAway": "away", "team": {"id": "...", "displayName": "..."}, "score": "72"}
                ]
              }]
            }
          ]
        }
        """
        games = []
        events = data.get("events", [])

        for event in events:
            try:
                competitions = event.get("competitions", [])
                if not competitions:
                    continue
                comp = competitions[0]

                # Status
                status_obj = comp.get("status", {}).get("type", {})
                status_name = status_obj.get("name", "STATUS_SCHEDULED")
                status = _espn_status_to_str(status_name)

                # Neutral site
                neutral_site = bool(comp.get("neutralSite", False))

                # Competitors
                competitors = comp.get("competitors", [])
                home_data = next(
                    (c for c in competitors if c.get("homeAway") == "home"), None
                )
                away_data = next(
                    (c for c in competitors if c.get("homeAway") == "away"), None
                )
                if not home_data or not away_data:
                    continue

                home_team_info = home_data.get("team", {})
                away_team_info = away_data.get("team", {})

                home_score_raw = home_data.get("score")
                away_score_raw = away_data.get("score")

                try:
                    home_score = float(home_score_raw) if home_score_raw else None
                    away_score = float(away_score_raw) if away_score_raw else None
                except (TypeError, ValueError):
                    home_score = None
                    away_score = None

                total_score = None
                if home_score is not None and away_score is not None:
                    total_score = home_score + away_score

                # Date (YYYY-MM-DD)
                raw_date = event.get("date", "")[:10]

                game = {
                    "game_id": f"espn_{event.get('id', comp.get('id', ''))}",
                    "espn_game_id": event.get("id", ""),
                    "date": raw_date,
                    "home_team": home_team_info.get("displayName", ""),
                    "home_team_id": f"espn_{home_team_info.get('id', '')}",
                    "home_team_abbreviation": home_team_info.get("abbreviation", ""),
                    "away_team": away_team_info.get("displayName", ""),
                    "away_team_id": f"espn_{away_team_info.get('id', '')}",
                    "away_team_abbreviation": away_team_info.get("abbreviation", ""),
                    "neutral_site": neutral_site,
                    "status": status,
                    "home_score": home_score,
                    "away_score": away_score,
                    "total_score": total_score,
                }
                games.append(game)

            except Exception as exc:
                logger.debug(f"Could not parse ESPN event: {exc}")
                continue

        return games

    # ── Public interface ──────────────────────────────────────────────────

    def get_todays_schedule(self) -> list[dict]:
        """Return today's CBB schedule."""
        today = date.today().strftime("%Y%m%d")
        return self._fetch_scoreboard(today)

    def get_schedule_by_date(self, date_str: str) -> list[dict]:
        """
        Return schedule for a given date.

        Args:
            date_str: Date in YYYY-MM-DD format.
        """
        espn_date = date_str.replace("-", "")
        return self._fetch_scoreboard(espn_date)

    def _fetch_scoreboard(self, espn_date: str) -> list[dict]:
        """Fetch and parse scoreboard for YYYYMMDD-formatted date."""
        logger.debug(f"Fetching ESPN scoreboard for {espn_date}")
        try:
            params = {"dates": espn_date, "groups": "50", "limit": "300"}
            data = self._get(_SCOREBOARD, params=params)
            games = self._parse_scoreboard(data)
            logger.info(f"ESPN scoreboard {espn_date}: found {len(games)} games.")
            return games
        except Exception as exc:
            logger.error(f"Failed to fetch ESPN scoreboard for {espn_date}: {exc}")
            return []

    def get_completed_games_by_date_range(
        self, start: str, end: str
    ) -> list[dict]:
        """
        Fetch all completed games between start and end dates (YYYY-MM-DD).
        Adds a small delay between requests to be polite to ESPN's servers.
        """
        results = []
        try:
            current = date.fromisoformat(start)
            end_d = date.fromisoformat(end)
        except ValueError:
            logger.error(f"Invalid date range: {start} → {end}")
            return results

        logger.info(f"Fetching ESPN schedule from {start} to {end}...")
        while current <= end_d:
            date_str = current.strftime("%Y-%m-%d")
            games = self.get_schedule_by_date(date_str)
            completed = [g for g in games if g.get("status") == "final"]
            results.extend(completed)
            current += timedelta(days=1)

        logger.info(
            f"Collected {len(results)} completed games from {start} to {end}."
        )
        return results


class ESPNStatsAdapter(BaseStatsAdapter):
    """
    Fetches team statistics from ESPN's public team stats API.
    """

    def __init__(self, config: dict):
        self.config = config
        espn_cfg = config.get("espn", {})
        self.request_delay: float = float(espn_cfg.get("request_delay", 0.5))
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (compatible; CBBTotalsModel/1.0)"
                )
            }
        )
        self._teams_cache: Optional[list[dict]] = None

    # ── Internal helpers ──────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
        reraise=True,
    )
    def _get(self, url: str, params: dict = None) -> dict:
        time.sleep(self.request_delay)
        resp = self._session.get(url, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def _espn_team_id(self, team_id: str) -> str:
        """Strip the 'espn_' prefix to get raw ESPN numeric ID."""
        return team_id.replace("espn_", "")

    # ── Public interface ──────────────────────────────────────────────────

    def get_team_stats(self, team_id: str, season: str) -> dict:
        """
        Fetch aggregate season stats for a team from ESPN.

        ESPN's team statistics endpoint:
        /teams/{id}/statistics

        Args:
            team_id: 'espn_123' format or raw ESPN numeric ID.
            season:  Season year string ('2024').

        Returns:
            Dict with stat keys matching TeamSeasonStats columns.
        """
        raw_id = self._espn_team_id(team_id)
        url = f"{_BASE}/teams/{raw_id}/statistics"
        params = {}
        if season:
            params["season"] = season

        logger.debug(f"Fetching ESPN stats for team {raw_id}, season {season}")
        try:
            data = self._get(url, params=params)
        except Exception as exc:
            logger.warning(f"ESPN stats fetch failed for team {raw_id}: {exc}")
            return {}

        return self._parse_team_stats(data, team_id, season)

    def _parse_team_stats(self, data: dict, team_id: str, season: str) -> dict:
        """
        Parse ESPN team statistics JSON into our stat schema.

        ESPN statistics response structure:
        {
          "team": {"id": "...", "displayName": "..."},
          "results": {
            "stats": {
              "categories": [
                {
                  "name": "offensive",
                  "stats": [{"name": "avgPoints", "value": 74.5}, ...]
                },
                ...
              ]
            }
          }
        }
        """
        stats: dict = {
            "team_id": team_id,
            "season": season,
        }

        try:
            results = data.get("results", {})
            stats_obj = results.get("stats", {})
            categories = stats_obj.get("categories", [])

            for cat in categories:
                cat_name = cat.get("name", "").lower()
                for stat_entry in cat.get("stats", []):
                    stat_name = stat_entry.get("name", "")
                    stat_value = stat_entry.get("value")

                    if stat_name in _STAT_MAP:
                        internal_key = _STAT_MAP[stat_name]
                        try:
                            stats[internal_key] = float(stat_value) if stat_value is not None else None
                        except (TypeError, ValueError):
                            stats[internal_key] = None

                    # Handle by category
                    if cat_name == "offensive":
                        if stat_name == "avgPoints":
                            stats["ppg"] = _safe_float(stat_value)
                        elif stat_name == "fieldGoalPct":
                            stats["efg_pct"] = _safe_float(stat_value)
                        elif stat_name == "threePointFieldGoalPct":
                            stats["three_p_pct"] = _safe_float(stat_value)
                        elif stat_name == "freeThrowPct":
                            stats["ft_pct"] = _safe_float(stat_value)
                        elif stat_name == "avgTurnovers":
                            stats["tov_per_game"] = _safe_float(stat_value)
                        elif stat_name == "avgAssists":
                            stats["avg_assists"] = _safe_float(stat_value)
                        elif stat_name == "avgRebounds":
                            stats["avg_rebounds"] = _safe_float(stat_value)
                        elif stat_name == "threePointFieldGoalsAttempted":
                            stats["three_pa_per_game"] = _safe_float(stat_value)

                    elif cat_name == "defensive":
                        if stat_name == "avgPoints":
                            stats["opp_ppg"] = _safe_float(stat_value)
                        elif stat_name == "avgRebounds":
                            stats["opp_avg_rebounds"] = _safe_float(stat_value)

            # Estimate raw_oe / raw_de from ppg if available
            if "ppg" in stats:
                stats["raw_oe"] = stats["ppg"]
            if "opp_ppg" in stats:
                stats["raw_de"] = stats["opp_ppg"]

        except Exception as exc:
            logger.warning(f"Error parsing ESPN team stats: {exc}")

        return stats

    def get_game_log(self, team_id: str, season: str) -> list[dict]:
        """
        Fetch per-game results for a team.

        ESPN's team schedule endpoint:
        /teams/{id}/schedule?season={year}
        """
        raw_id = self._espn_team_id(team_id)
        url = f"{_BASE}/teams/{raw_id}/schedule"
        params = {}
        if season:
            params["season"] = season

        logger.debug(f"Fetching ESPN game log for team {raw_id}, season {season}")
        try:
            data = self._get(url, params=params)
        except Exception as exc:
            logger.warning(f"ESPN game log fetch failed for team {raw_id}: {exc}")
            return []

        return self._parse_game_log(data, team_id)

    def _parse_game_log(self, data: dict, team_id: str) -> list[dict]:
        """
        Parse ESPN schedule JSON into a list of game results.

        ESPN schedule structure (simplified):
        {
          "events": [
            {
              "id": "...",
              "date": "2024-01-15T00:00Z",
              "competitions": [{
                "competitors": [
                  {"homeAway": "home", "team": {"id": "..."}, "score": "78",
                   "winner": true, "linescores": [...]},
                  {...}
                ],
                "status": {"type": {"name": "STATUS_FINAL"}}
              }]
            }
          ]
        }
        """
        games = []
        raw_id = team_id.replace("espn_", "")
        events = data.get("events", [])

        for event in events:
            try:
                comps = event.get("competitions", [])
                if not comps:
                    continue
                comp = comps[0]

                status_name = comp.get("status", {}).get("type", {}).get("name", "")
                if "FINAL" not in status_name.upper():
                    continue

                competitors = comp.get("competitors", [])
                # Find this team and the opponent
                team_comp = next(
                    (
                        c
                        for c in competitors
                        if c.get("team", {}).get("id") == raw_id
                    ),
                    None,
                )
                opp_comp = next(
                    (
                        c
                        for c in competitors
                        if c.get("team", {}).get("id") != raw_id
                    ),
                    None,
                )

                if not team_comp or not opp_comp:
                    continue

                home = team_comp.get("homeAway") == "home"
                neutral_site = bool(comp.get("neutralSite", False))

                score = _safe_float(team_comp.get("score"))
                opp_score = _safe_float(opp_comp.get("score"))
                total = (score + opp_score) if (score and opp_score) else None

                opp_team_info = opp_comp.get("team", {})

                game = {
                    "game_id": f"espn_{event.get('id', '')}",
                    "date": event.get("date", "")[:10],
                    "home": home,
                    "neutral_site": neutral_site,
                    "score": score,
                    "opp_score": opp_score,
                    "total": total,
                    "opponent_id": f"espn_{opp_team_info.get('id', '')}",
                    "opponent_name": opp_team_info.get("displayName", ""),
                    "winner": bool(team_comp.get("winner", False)),
                }
                games.append(game)

            except Exception as exc:
                logger.debug(f"Could not parse game log event: {exc}")
                continue

        # Sort ascending by date
        games.sort(key=lambda g: g.get("date", ""))
        return games

    def get_all_teams(self) -> list[dict]:
        """
        Return a list of all D1 NCAA basketball teams from ESPN.

        Uses the /teams endpoint with pagination support.
        """
        if self._teams_cache is not None:
            return self._teams_cache

        teams = []
        page = 1
        limit = 100

        logger.info("Fetching all D1 teams from ESPN...")
        while True:
            try:
                params = {"groups": "50", "limit": limit, "page": page}
                data = self._get(_TEAMS, params=params)

                page_teams = data.get("sports", [{}])[0] if data.get("sports") else {}
                if not page_teams:
                    # Try alternate structure
                    raw_teams = data.get("teams", [])
                else:
                    raw_teams = []
                    leagues = page_teams.get("leagues", [])
                    for league in leagues:
                        raw_teams.extend(league.get("teams", []))

                if not raw_teams:
                    break

                for team_wrapper in raw_teams:
                    team_info = team_wrapper.get("team", team_wrapper)
                    team = {
                        "team_id": f"espn_{team_info.get('id', '')}",
                        "espn_id": team_info.get("id", ""),
                        "team_name": team_info.get("displayName", ""),
                        "abbreviation": team_info.get("abbreviation", ""),
                        "conference": _extract_conference(team_info),
                        "is_active": True,
                    }
                    if team["espn_id"]:
                        teams.append(team)

                # Check pagination
                page_info = data.get("pageInfo", {})
                total_pages = page_info.get("totalPages", 1)
                if page >= total_pages:
                    break
                page += 1

            except Exception as exc:
                logger.warning(f"Error fetching teams page {page}: {exc}")
                break

        logger.info(f"Found {len(teams)} teams from ESPN.")
        self._teams_cache = teams
        return teams


# ── Utilities ─────────────────────────────────────────────────────────────────

def _safe_float(value) -> Optional[float]:
    """Convert to float, return None on failure."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _espn_status_to_str(espn_status: str) -> str:
    """Map ESPN status name to simple string."""
    s = espn_status.upper()
    if "FINAL" in s:
        return "final"
    if "IN_PROGRESS" in s or "HALFTIME" in s:
        return "live"
    if "SCHEDULED" in s or "PRE" in s:
        return "scheduled"
    if "POSTPONED" in s or "CANCELLED" in s or "CANCELED" in s:
        return "cancelled"
    return "scheduled"


def _extract_conference(team_info: dict) -> Optional[str]:
    """Try to extract conference name from ESPN team info."""
    groups = team_info.get("groups", [])
    if groups:
        for g in groups:
            if isinstance(g, dict):
                return g.get("name")
    return None

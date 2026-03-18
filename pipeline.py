#!/usr/bin/env python3
"""
WagerHub CBB Totals — Standalone Pipeline
Pulls schedule + ratings + odds, projects game totals, writes outputs/today.json.
No database needed. Runs automatically via GitHub Actions.
"""
from __future__ import annotations

import difflib
import json
import logging
import os
from datetime import date, datetime
from pathlib import Path

import requests
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).parent
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)
(OUTPUTS / "history").mkdir(exist_ok=True)
(ROOT / "logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(ROOT / "logs" / "pipeline.log", mode="a", encoding="utf-8"),
    ],
)
log = logging.getLogger("wagerhub")

# ── League constants ───────────────────────────────────────────────────────────
LEAGUE_AVG_TEMPO = 68.5
LEAGUE_AVG_PPP   = 1.04    # points per possession
LEAGUE_AVG_OE    = 105.0
HOME_ADVANTAGE   = 3.5     # pts added to home team


# ═══════════════════════════════════════════════════════════════════════════════
# DATA SOURCES
# ═══════════════════════════════════════════════════════════════════════════════

def get_torvik_ratings(season: int | None = None) -> dict[str, dict]:
    """
    Pull T-Rank ratings from barttorvik.com — no API key needed.
    Returns {TEAM_NAME_UPPER: {adj_oe, adj_de, adj_tempo, rank}}.
    """
    year = season or date.today().year
    url  = f"https://barttorvik.com/trank.php?year={year}&csv=1"
    log.info(f"Pulling Torvik ratings ({year})…")

    try:
        r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        from io import StringIO
        df = pd.read_csv(StringIO(r.text), header=None)
    except Exception as exc:
        log.warning(f"Torvik unavailable ({exc}) — projections will use league averages.")
        return {}

    ratings: dict[str, dict] = {}
    for _, row in df.iterrows():
        try:
            name = str(row[0]).strip()
            ratings[name.upper()] = {
                "adj_oe":    float(row[2]) if pd.notna(row[2]) else LEAGUE_AVG_OE,
                "adj_de":    float(row[3]) if pd.notna(row[3]) else LEAGUE_AVG_OE,
                "adj_tempo": float(row[4]) if pd.notna(row[4]) else LEAGUE_AVG_TEMPO,
                "rank":      int(row[1])   if pd.notna(row[1]) else 200,
            }
        except (ValueError, IndexError):
            continue

    log.info(f"Torvik: {len(ratings)} teams loaded")
    return ratings


def get_espn_schedule(date_str: str) -> list[dict]:
    """Pull today's NCAA Men's Basketball schedule from ESPN (no key needed)."""
    url = (
        "https://site.api.espn.com/apis/site/v2/sports/basketball"
        f"/mens-college-basketball/scoreboard?dates={date_str.replace('-','')}&limit=200"
    )
    log.info(f"Pulling ESPN schedule for {date_str}…")

    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        log.error(f"ESPN schedule failed: {exc}")
        return []

    games: list[dict] = []
    for event in data.get("events", []):
        try:
            comp    = event["competitions"][0]
            by_side = {c["homeAway"]: c for c in comp["competitors"]}
            home    = by_side.get("home", {})
            away    = by_side.get("away", {})

            raw_status = event.get("status", {}).get("type", {}).get("name", "").lower()
            if "post" in raw_status:
                status = "final"
            elif "in" in raw_status:
                status = "live"
            else:
                status = "scheduled"

            # Parse kick-off time → rough Eastern display
            start = event.get("date", "")
            try:
                from datetime import timezone, timedelta
                dt_utc  = datetime.fromisoformat(start.replace("Z", "+00:00"))
                dt_est  = dt_utc - timedelta(hours=5)
                hour    = dt_est.hour
                minute  = dt_est.minute
                suffix  = "AM" if hour < 12 else "PM"
                hour12  = hour % 12 or 12
                game_time = f"{hour12}:{minute:02d} {suffix} ET"
            except Exception:
                game_time = ""

            games.append({
                "game_id":      event["id"],
                "game_date":    date_str,
                "home_team":    home.get("team", {}).get("displayName", ""),
                "away_team":    away.get("team", {}).get("displayName", ""),
                "neutral_site": comp.get("neutralSite", False),
                "status":       status,
                "game_time":    game_time,
                "home_score":   int(home.get("score", 0)) if status in ("final", "live") else None,
                "away_score":   int(away.get("score", 0)) if status in ("final", "live") else None,
            })
        except (KeyError, IndexError, TypeError):
            continue

    log.info(f"ESPN: {len(games)} games")
    return games


def get_odds(api_key: str) -> list[dict]:
    """Pull NCAAB totals from The Odds API."""
    if not api_key:
        log.warning("THE_ODDS_API_KEY not set — odds will be missing from output.")
        return []

    url = (
        "https://api.the-odds-api.com/v4/sports/basketball_ncaab/odds/"
        f"?apiKey={api_key}&regions=us&markets=totals&oddsFormat=american"
        "&bookmakers=draftkings,fanduel,betmgm,pinnacle,williamhill_us"
    )
    log.info("Pulling odds from The Odds API…")

    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        data = r.json()
        log.info(f"Odds API: {r.headers.get('x-requests-remaining', '?')} requests remaining this month")
    except Exception as exc:
        log.error(f"Odds API failed: {exc}")
        return []

    lines: list[dict] = []
    for game in data:
        for bm in game.get("bookmakers", []):
            for mkt in bm.get("markets", []):
                if mkt.get("key") != "totals":
                    continue
                outcomes = {o["name"]: o for o in mkt.get("outcomes", [])}
                over = outcomes.get("Over", {})
                if not over.get("point"):
                    continue
                under = outcomes.get("Under", {})
                lines.append({
                    "home_team":    game.get("home_team", ""),
                    "away_team":    game.get("away_team", ""),
                    "sportsbook":   bm.get("key", ""),
                    "market_total": float(over["point"]),
                    "over_odds":    int(over.get("price", -110)),
                    "under_odds":   int(under.get("price", -110)),
                })

    log.info(f"Odds: {len(lines)} lines across {len(data)} games")
    return lines


# ═══════════════════════════════════════════════════════════════════════════════
# MATCHING
# ═══════════════════════════════════════════════════════════════════════════════

def _sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a.upper(), b.upper()).ratio()


def find_torvik(team: str, ratings: dict) -> dict | None:
    upper = team.upper()
    if upper in ratings:
        return ratings[upper]
    best, best_k = 0.0, None
    for k in ratings:
        s = _sim(team, k)
        if s > best:
            best, best_k = s, k
    return ratings[best_k] if best_k and best >= 0.60 else None


def find_odds_for_game(game: dict, all_lines: list[dict]) -> list[dict]:
    """Return all odds lines that match this ESPN game."""
    matches = []
    for line in all_lines:
        hs = _sim(game["home_team"], line["home_team"])
        as_ = _sim(game["away_team"], line["away_team"])
        if (hs + as_) / 2 >= 0.55:
            matches.append(line)
    return matches


def consensus_line(matched_odds: list[dict]) -> dict | None:
    if not matched_odds:
        return None
    totals = [o["market_total"] for o in matched_odds]
    avg    = sum(totals) / len(totals)
    # Round to nearest 0.5
    avg    = round(avg * 2) / 2

    pref   = ["draftkings", "fanduel", "betmgm", "pinnacle"]
    base   = next((o for p in pref for o in matched_odds if o["sportsbook"] == p), matched_odds[0])

    return {
        "market_total": avg,
        "over_odds":    base["over_odds"],
        "under_odds":   base["under_odds"],
        "sportsbook":   "Consensus" if len(totals) > 1 else base["sportsbook"].replace("_", " ").title(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PROJECTION MODEL
# ═══════════════════════════════════════════════════════════════════════════════

def project(home: str, away: str, ratings: dict, neutral: bool = False) -> dict:
    """
    Possessions × efficiency baseline model.
    home_ppp = (home_adj_oe / lg_oe) * (away_adj_de / lg_de) * lg_ppp
    """
    h = find_torvik(home, ratings) or {"adj_oe": LEAGUE_AVG_OE, "adj_de": LEAGUE_AVG_OE, "adj_tempo": LEAGUE_AVG_TEMPO}
    a = find_torvik(away, ratings) or {"adj_oe": LEAGUE_AVG_OE, "adj_de": LEAGUE_AVG_OE, "adj_tempo": LEAGUE_AVG_TEMPO}

    # Possessions — blend of each team's tempo, slight regression to mean
    raw_poss = (h["adj_tempo"] + a["adj_tempo"]) / 2
    poss     = raw_poss * 0.88 + LEAGUE_AVG_TEMPO * 0.12

    lg = LEAGUE_AVG_OE
    home_ppp = (h["adj_oe"] / lg) * (a["adj_de"] / lg) * LEAGUE_AVG_PPP
    away_ppp = (a["adj_oe"] / lg) * (h["adj_de"] / lg) * LEAGUE_AVG_PPP

    home_pts = home_ppp * poss
    away_pts = away_ppp * poss

    if not neutral:
        home_pts += HOME_ADVANTAGE / 2
        away_pts -= HOME_ADVANTAGE / 2

    total = home_pts + away_pts

    # Confidence: higher when both teams have distinct, well-defined ratings
    h_found = find_torvik(home, ratings) is not None
    a_found = find_torvik(away, ratings) is not None
    if h_found and a_found:
        spread = abs(h["adj_oe"] - h["adj_de"]) + abs(a["adj_oe"] - a["adj_de"])
        conf   = min(0.88, 0.44 + spread / 55)
    else:
        conf   = 0.28

    return {
        "ensemble_total":        round(total, 1),
        "predicted_home_score":  round(home_pts, 1),
        "predicted_away_score":  round(away_pts, 1),
        "predicted_possessions": round(poss, 1),
        "confidence_score":      round(conf, 2),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE ORCHESTRATION
# ═══════════════════════════════════════════════════════════════════════════════

def run(date_str: str | None = None) -> list[dict]:
    today    = date_str or date.today().strftime("%Y-%m-%d")
    api_key  = os.environ.get("THE_ODDS_API_KEY", "")
    now_utc  = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    log.info(f"{'='*55}")
    log.info(f"WagerHub CBB Pipeline  —  {today}")
    log.info(f"{'='*55}")

    ratings  = get_torvik_ratings()
    schedule = get_espn_schedule(today)
    odds     = get_odds(api_key)

    if not schedule:
        log.warning("No games on schedule. Exiting.")
        return []

    results: list[dict] = []
    for game in schedule:
        if game["status"] == "final":
            continue  # don't include completed games in today's slate

        proj        = project(game["home_team"], game["away_team"], ratings, game.get("neutral_site", False))
        matched     = find_odds_for_game(game, odds)
        market_line = consensus_line(matched)

        mkt   = market_line["market_total"] if market_line else None
        mdl   = proj["ensemble_total"]
        diff  = round(mdl - mkt, 1) if mkt is not None else None
        edge  = ("OVER" if diff > 0 else "UNDER") if diff is not None else None

        results.append({
            # Game info
            "game_id":               game["game_id"],
            "game_date":             today,
            "away_team":             game["away_team"],
            "home_team":             game["home_team"],
            "game_time":             game.get("game_time", ""),
            "neutral_site":          game.get("neutral_site", False),
            "status":                game["status"],
            # Projection
            "ensemble_total":        mdl,
            "predicted_away_score":  proj["predicted_away_score"],
            "predicted_home_score":  proj["predicted_home_score"],
            "predicted_possessions": proj["predicted_possessions"],
            "confidence_score":      proj["confidence_score"],
            # Market
            "market_total":          mkt,
            "sportsbook":            market_line["sportsbook"] if market_line else None,
            "over_odds":             market_line["over_odds"]  if market_line else None,
            "under_odds":            market_line["under_odds"] if market_line else None,
            # Edge
            "differential":          diff,
            "abs_differential":      abs(diff) if diff is not None else None,
            "edge_side":             edge,
            # Meta
            "run_timestamp":         now_utc,
        })

    # Sort: biggest edge first
    results.sort(key=lambda r: r["abs_differential"] or 0, reverse=True)

    log.info(f"Results: {len(results)} games projected")
    log.info(f"  With market lines : {sum(1 for r in results if r['market_total'])}")
    log.info(f"  Strong edges (≥5) : {sum(1 for r in results if (r['abs_differential'] or 0) >= 5)}")
    return results


def save(results: list[dict], date_str: str) -> None:
    payload = {
        "generated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "date":         date_str,
        "game_count":   len(results),
        "games":        results,
    }
    for path in [OUTPUTS / "today.json", OUTPUTS / "history" / f"{date_str}.json"]:
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        log.info(f"Saved  {path}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="WagerHub CBB pipeline")
    ap.add_argument("--date", help="YYYY-MM-DD (default: today)")
    args = ap.parse_args()

    d = args.date or date.today().strftime("%Y-%m-%d")
    results = run(d)
    save(results, d)
    log.info("Done.")

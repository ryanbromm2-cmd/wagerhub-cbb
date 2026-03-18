"""
WagerHub — CBB Totals Dashboard
Run:  streamlit run dashboard/app.py
"""
from __future__ import annotations

import sys
from datetime import date, datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import yaml

REFRESH_INTERVAL_MS = 5 * 60 * 1000  # auto-refresh every 5 minutes

# ── Must be first Streamlit call ──────────────────────────────────────────────
st.set_page_config(
    page_title="WagerHub · CBB Totals",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"Get Help": None, "Report a bug": None, "About": "WagerHub CBB Totals Model"},
)

# ── CSS ───────────────────────────────────────────────────────────────────────
DARK_CSS = """
<style>
/* ── Layout ─────────────────────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem; padding-bottom: 2rem; }

/* ── Sidebar ─────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #0e1118 !important;
    border-right: 1px solid #1e2433;
}
[data-testid="stSidebar"] .stRadio label {
    font-size: 14px;
    padding: 4px 0;
}

/* ── WagerHub brand bar ───────────────────────────────────── */
.wh-topbar {
    display: flex;
    align-items: baseline;
    gap: 10px;
    padding: 4px 0 18px;
    border-bottom: 2px solid #f59e0b;
    margin-bottom: 22px;
}
.wh-logo {
    font-size: 26px;
    font-weight: 900;
    letter-spacing: -0.5px;
    color: #f59e0b;
    line-height: 1;
}
.wh-logo span { color: #e2e8f0; }
.wh-sub {
    font-size: 13px;
    color: #64748b;
    letter-spacing: 1px;
    text-transform: uppercase;
    padding-left: 4px;
}
.wh-date-badge {
    margin-left: auto;
    background: #12151e;
    border: 1px solid #1e2433;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 13px;
    color: #94a3b8;
}

/* ── Demo banner ─────────────────────────────────────────── */
.demo-banner {
    background: linear-gradient(90deg, #1c1a08, #221f08);
    border: 1px solid #854d0e;
    border-left: 4px solid #f59e0b;
    border-radius: 8px;
    padding: 10px 16px;
    margin-bottom: 18px;
    font-size: 13px;
    color: #fde68a;
}

/* ── Stat cards ──────────────────────────────────────────── */
.stat-card {
    background: #12151e;
    border: 1px solid #1e2433;
    border-radius: 10px;
    padding: 18px 20px;
    text-align: center;
}
.stat-card-value {
    font-size: 30px;
    font-weight: 800;
    line-height: 1.1;
    margin-bottom: 4px;
}
.stat-card-label {
    font-size: 11px;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}
.stat-val-gold   { color: #f59e0b; }
.stat-val-green  { color: #4ade80; }
.stat-val-red    { color: #f87171; }

/* ── Games table ─────────────────────────────────────────── */
.games-wrap {
    background: #12151e;
    border: 1px solid #1e2433;
    border-radius: 10px;
    overflow: hidden;
    margin-top: 4px;
}
.games-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13.5px;
}
.games-table th {
    background: #0e1118;
    color: #64748b;
    font-size: 10.5px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.9px;
    padding: 11px 14px;
    text-align: left;
    border-bottom: 1px solid #1e2433;
    white-space: nowrap;
}
.games-table th.right { text-align: right; }
.games-table td {
    padding: 13px 14px;
    vertical-align: middle;
    border-bottom: 1px solid #181d28;
}
.games-table td.right { text-align: right; }
.games-table tr:last-child td { border-bottom: none; }
.games-table tr:hover td { background: #161b26; }

/* row accent by edge type */
.row-strong-over  td:first-child { border-left: 3px solid #22c55e; }
.row-medium-over  td:first-child { border-left: 3px solid #4ade80; }
.row-mild-over    td:first-child { border-left: 3px solid #86efac; }
.row-strong-under td:first-child { border-left: 3px solid #ef4444; }
.row-medium-under td:first-child { border-left: 3px solid #f87171; }
.row-mild-under   td:first-child { border-left: 3px solid #fca5a5; }
.row-neutral      td:first-child { border-left: 3px solid #374151; }

/* rank badge */
.rank {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    background: #1e2433;
    color: #64748b;
    border-radius: 50%;
    width: 22px;
    height: 22px;
    font-size: 11px;
    font-weight: 700;
}

/* team matchup */
.matchup { white-space: nowrap; }
.team-away { font-weight: 600; color: #cbd5e1; }
.team-at   { color: #374151; font-size: 11px; margin: 0 4px; }
.team-home { font-weight: 600; color: #e2e8f0; }
.game-time { font-size: 11px; color: #475569; margin-top: 2px; }

/* totals */
.mkt-total  { font-size: 15px; font-weight: 700; color: #e2e8f0; }
.mdl-total  { font-size: 15px; font-weight: 700; color: #93c5fd; }

/* differential */
.diff-strong-over  { font-size: 16px; font-weight: 800; color: #4ade80; }
.diff-medium-over  { font-size: 15px; font-weight: 700; color: #86efac; }
.diff-mild-over    { font-size: 14px; font-weight: 600; color: #bbf7d0; }
.diff-strong-under { font-size: 16px; font-weight: 800; color: #f87171; }
.diff-medium-under { font-size: 15px; font-weight: 700; color: #fca5a5; }
.diff-mild-under   { font-size: 14px; font-weight: 600; color: #fecaca; }
.diff-none         { font-size: 13px; color: #374151; }

/* edge badge */
.badge-over {
    background: #052e16;
    color: #4ade80;
    border: 1px solid #166534;
    border-radius: 5px;
    padding: 3px 9px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.5px;
    white-space: nowrap;
}
.badge-under {
    background: #2d0a0a;
    color: #f87171;
    border: 1px solid #7f1d1d;
    border-radius: 5px;
    padding: 3px 9px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.5px;
    white-space: nowrap;
}
.badge-none {
    background: #1e2433;
    color: #475569;
    border: 1px solid #2d3748;
    border-radius: 5px;
    padding: 3px 9px;
    font-size: 11px;
    white-space: nowrap;
}

/* confidence bar */
.conf-wrap {
    display: flex;
    align-items: center;
    gap: 7px;
    white-space: nowrap;
}
.conf-track {
    background: #1e2433;
    border-radius: 4px;
    height: 6px;
    width: 60px;
    overflow: hidden;
    display: inline-block;
}
.conf-fill {
    height: 100%;
    border-radius: 4px;
}
.conf-label {
    font-size: 11px;
    color: #64748b;
}

/* scores sub-cell */
.scores {
    font-size: 11.5px;
    color: #64748b;
    white-space: nowrap;
}

/* sportsbook pill */
.book-pill {
    background: #1e2433;
    border-radius: 20px;
    padding: 2px 9px;
    font-size: 11px;
    color: #94a3b8;
    white-space: nowrap;
}

/* no-games state */
.no-games {
    text-align: center;
    padding: 60px 20px;
    color: #475569;
}
.no-games h3 { font-size: 18px; color: #64748b; margin-bottom: 8px; }
.no-games p  { font-size: 14px; }

/* Section header */
.section-header {
    font-size: 13px;
    font-weight: 700;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin: 24px 0 10px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #1e2433;
}

/* Filter bar */
.filter-bar {
    background: #12151e;
    border: 1px solid #1e2433;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 16px;
}

/* ── Responsive ──────────────────────────────────────────── */
@media (max-width: 768px) {
    .games-table { font-size: 12px; }
    .games-table th, .games-table td { padding: 9px 8px; }
    .conf-track { width: 40px; }
    .wh-logo { font-size: 20px; }
}
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)


# ── Config ────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_config() -> dict:
    cfg_path = PROJECT_ROOT / "config" / "config.yaml"
    if cfg_path.exists():
        with open(cfg_path) as f:
            return yaml.safe_load(f)
    return {}


@st.cache_resource
def get_db():
    """Try to connect to the database; return None silently on failure."""
    try:
        from src.utils.db import DatabaseManager
        cfg = load_config()
        db = DatabaseManager(cfg)
        db.init_db()
        return db
    except Exception:
        return None


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_slate(game_date: str) -> tuple[pd.DataFrame, bool]:
    """
    Returns (df, is_demo). Falls back to demo data when DB is empty or unavailable.
    """
    db = get_db()
    df = pd.DataFrame()

    if db:
        try:
            projs = db.get_todays_projections(game_date) or []
            if projs:
                df = pd.DataFrame(projs)
                odds_raw = db.get_odds_for_date(game_date) or []
                if odds_raw:
                    odds_df = pd.DataFrame(odds_raw)
                    odds_df = (
                        odds_df
                        .sort_values("sportsbook", key=lambda s: s.map({"consensus": 0}).fillna(1))
                        .drop_duplicates("game_id")
                    )
                    merge_cols = [c for c in ["game_id", "market_total", "over_odds", "under_odds", "sportsbook"] if c in odds_df.columns]
                    df = df.merge(odds_df[merge_cols], on="game_id", how="left")
                if "ensemble_total" in df.columns and "market_total" in df.columns:
                    df["differential"] = df["ensemble_total"] - df["market_total"]
                    df["abs_differential"] = df["differential"].abs()
                    df["edge_side"] = df["differential"].apply(
                        lambda d: "OVER" if d > 0 else ("UNDER" if d < 0 else "PUSH") if pd.notna(d) else None
                    )
                if not df.empty:
                    return df, False
        except Exception:
            pass

    # Fall back to demo data
    from dashboard.demo_data import get_demo_slate
    return pd.DataFrame(get_demo_slate(game_date)), True


@st.cache_data(ttl=600)
def load_line_history(game_id: str) -> pd.DataFrame:
    db = get_db()
    if db:
        try:
            data = db.get_line_history(game_id)
            if data:
                return pd.DataFrame(data)
        except Exception:
            pass
    from dashboard.demo_data import get_demo_line_history
    rows = get_demo_line_history(game_id)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


@st.cache_data(ttl=600)
def load_historical() -> pd.DataFrame:
    db = get_db()
    if db:
        try:
            data = db.get_backtest_results() or []
            if data:
                return pd.DataFrame(data), False
        except Exception:
            pass
    from dashboard.demo_data import get_demo_historical
    return pd.DataFrame(get_demo_historical()), True


@st.cache_data(ttl=3600)
def load_feature_importance() -> pd.DataFrame:
    try:
        from src.models.ml_model import MLModelTrainer
        cfg = load_config()
        trainer = MLModelTrainer(cfg)
        trainer.load_models(str(PROJECT_ROOT / "models"))
        return trainer.get_feature_importance()
    except Exception:
        return pd.DataFrame()


# ── Rendering helpers ─────────────────────────────────────────────────────────

def _row_class(edge_side: str | None, abs_diff: float | None) -> str:
    if not edge_side or edge_side == "PUSH" or abs_diff is None:
        return "row-neutral"
    d = float(abs_diff)
    side = edge_side.upper()
    if d >= 6:
        return f"row-strong-{'over' if side == 'OVER' else 'under'}"
    elif d >= 3.5:
        return f"row-medium-{'over' if side == 'OVER' else 'under'}"
    else:
        return f"row-mild-{'over' if side == 'OVER' else 'under'}"


def _diff_class(edge_side: str | None, abs_diff: float | None) -> str:
    if not edge_side or edge_side == "PUSH" or abs_diff is None:
        return "diff-none"
    d = float(abs_diff)
    side = edge_side.upper()
    if d >= 6:
        return f"diff-strong-{'over' if side == 'OVER' else 'under'}"
    elif d >= 3.5:
        return f"diff-medium-{'over' if side == 'OVER' else 'under'}"
    else:
        return f"diff-mild-{'over' if side == 'OVER' else 'under'}"


def _conf_color(score: float) -> str:
    if score >= 0.75:
        return "#4ade80"
    elif score >= 0.55:
        return "#facc15"
    else:
        return "#f97316"


def _conf_label(score: float) -> str:
    if score >= 0.75:
        return "High"
    elif score >= 0.55:
        return "Med"
    else:
        return "Low"


def _fmt_diff(diff: float | None, edge_side: str | None) -> str:
    if diff is None or not isinstance(diff, (int, float)):
        return "—"
    sign = "+" if diff > 0 else ""
    return f"{sign}{diff:.1f}"


def render_game_table(df: pd.DataFrame) -> str:
    """Build a fully styled HTML table for the games slate."""
    rows_html = ""
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        # Raw values
        away      = row.get("away_team") or row.get("away_team_id") or "Away"
        home      = row.get("home_team") or row.get("home_team_id") or "Home"
        mkt       = row.get("market_total")
        mdl       = row.get("ensemble_total")
        diff      = row.get("differential")
        abs_diff  = row.get("abs_differential") or (abs(diff) if isinstance(diff, float) else None)
        edge      = row.get("edge_side") or ""
        conf      = row.get("confidence_score") or 0.0
        poss      = row.get("predicted_possessions")
        away_pts  = row.get("predicted_away_score")
        home_pts  = row.get("predicted_home_score")
        book      = row.get("sportsbook") or "—"
        gtime     = row.get("game_time") or ""
        game_id   = row.get("game_id", "")

        # Formatted strings
        mkt_str   = f"{mkt:.1f}"  if isinstance(mkt, float)  else "—"
        mdl_str   = f"{mdl:.1f}"  if isinstance(mdl, float)  else "—"
        diff_str  = _fmt_diff(diff, edge)
        poss_str  = f"{poss:.1f}" if isinstance(poss, float) else "—"

        if isinstance(away_pts, float) and isinstance(home_pts, float):
            scores_str = f"{away_pts:.1f} – {home_pts:.1f}"
        else:
            scores_str = "—"

        conf_pct  = int(conf * 100) if isinstance(conf, float) else 0
        conf_col  = _conf_color(conf)
        conf_lbl  = _conf_label(conf)

        # CSS classes
        row_cls   = _row_class(edge, abs_diff)
        diff_cls  = _diff_class(edge, abs_diff)
        badge_cls = "badge-over" if edge == "OVER" else ("badge-under" if edge == "UNDER" else "badge-none")
        badge_txt = edge if edge else "—"

        rows_html += f"""
<tr class="{row_cls}" data-gameid="{game_id}">
  <td><span class="rank">{i}</span></td>
  <td class="matchup">
    <div>
      <span class="team-away">{away}</span>
      <span class="team-at">@</span>
      <span class="team-home">{home}</span>
    </div>
    {f'<div class="game-time">{gtime}</div>' if gtime else ""}
  </td>
  <td class="right"><span class="mkt-total">{mkt_str}</span></td>
  <td class="right"><span class="mdl-total">{mdl_str}</span></td>
  <td class="right"><span class="{diff_cls}">{diff_str}</span></td>
  <td><span class="{badge_cls}">{badge_txt}</span></td>
  <td>
    <div class="conf-wrap">
      <div class="conf-track">
        <div class="conf-fill" style="width:{conf_pct}%;background:{conf_col};"></div>
      </div>
      <span class="conf-label">{conf_lbl} {conf_pct}%</span>
    </div>
  </td>
  <td><span class="scores">{scores_str}</span></td>
  <td class="right"><span class="scores">{poss_str}</span></td>
  <td><span class="book-pill">{book}</span></td>
</tr>"""

    if not rows_html:
        rows_html = """
<tr><td colspan="10">
  <div class="no-games">
    <h3>No games match your filters</h3>
    <p>Try adjusting the filters above.</p>
  </div>
</td></tr>"""

    return f"""
<div class="games-wrap">
<table class="games-table">
<thead>
  <tr>
    <th>#</th>
    <th>Matchup</th>
    <th class="right">Mkt Total</th>
    <th class="right">Model Total</th>
    <th class="right">Differential</th>
    <th>Edge</th>
    <th>Confidence</th>
    <th>Proj Scores</th>
    <th class="right">Poss</th>
    <th>Book</th>
  </tr>
</thead>
<tbody>
{rows_html}
</tbody>
</table>
</div>"""


def wh_header(subtitle: str = "CBB Totals Model", show_date: bool = True):
    date_badge = (
        f'<span class="wh-date-badge">📅 {date.today().strftime("%A, %b %d %Y")}</span>'
        if show_date else ""
    )
    st.markdown(f"""
<div class="wh-topbar">
  <span class="wh-logo">Wager<span>Hub</span></span>
  <span class="wh-sub">{subtitle}</span>
  {date_badge}
</div>""", unsafe_allow_html=True)


def demo_banner():
    st.markdown("""
<div class="demo-banner">
  ⚡ <strong>Demo Mode</strong> — Showing sample data. Run <code>python main.py run</code> to populate
  with today's real projections and live odds.
</div>""", unsafe_allow_html=True)


def section_header(text: str):
    st.markdown(f'<div class="section-header">{text}</div>', unsafe_allow_html=True)


# ── Page: Today's Slate ───────────────────────────────────────────────────────

def page_todays_slate():
    wh_header("CBB Totals Model")

    # Date selector
    col_date, col_refresh, _ = st.columns([2, 1, 5])
    with col_date:
        selected_date = st.date_input("Date", value=date.today(), label_visibility="collapsed")
    with col_refresh:
        if st.button("↺ Refresh", help="Clear cache and reload data"):
            st.cache_data.clear()
            st.rerun()

    date_str = selected_date.strftime("%Y-%m-%d")
    df, is_demo = load_slate(date_str)

    if is_demo:
        demo_banner()

    if df.empty:
        st.markdown("""
<div class="no-games">
  <h3>No games scheduled</h3>
  <p>There may be no NCAA games today, or the pipeline hasn't run yet.</p>
</div>""", unsafe_allow_html=True)
        return

    # Ensure columns
    for col in ["abs_differential", "differential", "edge_side", "confidence_score"]:
        if col not in df.columns:
            df[col] = None

    # ── Stat cards ────────────────────────────────────────────────────────
    total_games = len(df)
    has_edges = df["abs_differential"].notna().any()

    best_over = df[df["edge_side"] == "OVER"].nlargest(1, "abs_differential") if has_edges else pd.DataFrame()
    best_under = df[df["edge_side"] == "UNDER"].nlargest(1, "abs_differential") if has_edges else pd.DataFrame()
    strong_edges = df[df["abs_differential"].fillna(0) >= 5] if has_edges else pd.DataFrame()

    best_over_str  = f"+{best_over.iloc[0]['abs_differential']:.1f}" if not best_over.empty else "—"
    best_under_str = f"-{best_under.iloc[0]['abs_differential']:.1f}" if not best_under.empty else "—"
    strong_count   = len(strong_edges)

    c1, c2, c3, c4 = st.columns(4)
    for col, value, label, css_class in [
        (c1, str(total_games),    "Games Today",       "stat-val-gold"),
        (c2, best_over_str,       "Best Over Edge",    "stat-val-green"),
        (c3, best_under_str,      "Best Under Edge",   "stat-val-red"),
        (c4, str(strong_count),   "Strong Edges (≥5)", "stat-val-gold"),
    ]:
        col.markdown(f"""
<div class="stat-card">
  <div class="stat-card-value {css_class}">{value}</div>
  <div class="stat-card-label">{label}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Filters ───────────────────────────────────────────────────────────
    section_header("Filters")

    fc1, fc2, fc3 = st.columns([2, 2, 3])
    with fc1:
        edge_filter = st.selectbox(
            "Edge Direction",
            ["All Edges", "OVER only", "UNDER only"],
            label_visibility="collapsed",
        )
    with fc2:
        conf_filter = st.selectbox(
            "Confidence",
            ["All Confidence", "High only (≥75%)", "Medium+ (≥55%)", "Low (<55%)"],
            label_visibility="collapsed",
        )
    with fc3:
        search = st.text_input("Search teams...", placeholder="e.g. Duke, Kentucky…", label_visibility="collapsed")

    min_edge = st.slider("Minimum edge (pts)", 0.0, 12.0, 0.0, 0.5, label_visibility="collapsed",
                         help="Filter to only show games where |Model − Market| ≥ this value")

    # Apply filters
    filtered = df.copy()

    if edge_filter == "OVER only":
        filtered = filtered[filtered["edge_side"] == "OVER"]
    elif edge_filter == "UNDER only":
        filtered = filtered[filtered["edge_side"] == "UNDER"]

    if conf_filter == "High only (≥75%)":
        filtered = filtered[filtered["confidence_score"].fillna(0) >= 0.75]
    elif conf_filter == "Medium+ (≥55%)":
        filtered = filtered[filtered["confidence_score"].fillna(0) >= 0.55]
    elif conf_filter == "Low (<55%)":
        filtered = filtered[filtered["confidence_score"].fillna(1) < 0.55]

    if min_edge > 0:
        filtered = filtered[filtered["abs_differential"].fillna(0) >= min_edge]

    if search.strip():
        q = search.strip().lower()
        away_col = "away_team" if "away_team" in filtered.columns else "away_team_id"
        home_col = "home_team" if "home_team" in filtered.columns else "home_team_id"
        mask = (
            filtered[away_col].str.lower().str.contains(q, na=False) |
            filtered[home_col].str.lower().str.contains(q, na=False)
        )
        filtered = filtered[mask]

    # Sort by abs_differential
    if "abs_differential" in filtered.columns:
        filtered = filtered.sort_values("abs_differential", ascending=False, na_position="last")

    # ── Game table ────────────────────────────────────────────────────────
    section_header(f"Today's Slate — {len(filtered)} game{'s' if len(filtered) != 1 else ''}")
    table_html = render_game_table(filtered)
    st.markdown(table_html, unsafe_allow_html=True)

    # Last updated
    if "run_timestamp" in df.columns:
        ts = pd.to_datetime(df["run_timestamp"], errors="coerce").max()
        if pd.notna(ts):
            st.markdown(
                f'<p class="last-updated" style="margin-top:8px;text-align:right;">'
                f'Last model run: {ts.strftime("%I:%M %p ET")}</p>',
                unsafe_allow_html=True
            )


# ── Page: Best Edges ──────────────────────────────────────────────────────────

def page_best_edges():
    wh_header("Best Edges", show_date=True)

    col1, col2, col3 = st.columns([2, 2, 2])
    with col1:
        selected_date = st.date_input("Date", value=date.today())
    with col2:
        min_edge = st.slider("Min edge (pts)", 0.0, 12.0, 3.0, 0.5)
    with col3:
        direction = st.selectbox("Direction", ["Both", "OVER", "UNDER"])

    date_str = selected_date.strftime("%Y-%m-%d")
    df, is_demo = load_slate(date_str)
    if is_demo:
        demo_banner()

    if df.empty or "abs_differential" not in df.columns:
        st.info("No data for this date.")
        return

    filtered = df[df["abs_differential"].fillna(0) >= min_edge].copy()
    if direction != "Both":
        filtered = filtered[filtered["edge_side"] == direction]
    filtered = filtered.sort_values("abs_differential", ascending=False)

    if filtered.empty:
        st.info(f"No edges ≥ {min_edge} pts found.")
        return

    # Over/Under split
    over_df  = filtered[filtered["edge_side"] == "OVER"]
    under_df = filtered[filtered["edge_side"] == "UNDER"]

    col_o, col_u = st.columns(2)

    with col_o:
        section_header(f"▲ OVER Edges ({len(over_df)})")
        if over_df.empty:
            st.markdown('<p style="color:#475569;font-size:13px;">No OVER edges meet the filter.</p>', unsafe_allow_html=True)
        else:
            st.markdown(render_game_table(over_df.head(10)), unsafe_allow_html=True)

    with col_u:
        section_header(f"▼ UNDER Edges ({len(under_df)})")
        if under_df.empty:
            st.markdown('<p style="color:#475569;font-size:13px;">No UNDER edges meet the filter.</p>', unsafe_allow_html=True)
        else:
            st.markdown(render_game_table(under_df.head(10)), unsafe_allow_html=True)

    # Bar chart
    section_header("Edge Chart")
    chart_df = filtered.copy()
    away_col = "away_team" if "away_team" in chart_df.columns else "away_team_id"
    home_col = "home_team" if "home_team" in chart_df.columns else "home_team_id"
    chart_df["game"] = chart_df[away_col].fillna("?") + " @ " + chart_df[home_col].fillna("?")

    fig = go.Figure()
    for side, color, bg in [("OVER", "#4ade80", "#052e16"), ("UNDER", "#f87171", "#2d0a0a")]:
        sub = chart_df[chart_df["edge_side"] == side].sort_values("abs_differential")
        if not sub.empty:
            fig.add_trace(go.Bar(
                x=sub["abs_differential"],
                y=sub["game"],
                orientation="h",
                name=side,
                marker_color=color,
                text=sub["abs_differential"].apply(lambda v: f"{v:.1f}"),
                textposition="outside",
                textfont=dict(color=color, size=12),
            ))
    fig.update_layout(
        paper_bgcolor="#12151e",
        plot_bgcolor="#12151e",
        font=dict(color="#94a3b8", size=12),
        xaxis=dict(gridcolor="#1e2433", title="Edge (pts)"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        legend=dict(bgcolor="#12151e", bordercolor="#1e2433"),
        barmode="overlay",
        height=max(300, len(chart_df) * 38),
        margin=dict(l=10, r=60, t=20, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Page: Line Movement ───────────────────────────────────────────────────────

def page_line_movement():
    wh_header("Line Movement", show_date=False)

    selected_date = st.date_input("Date", value=date.today())
    date_str = selected_date.strftime("%Y-%m-%d")
    df, is_demo = load_slate(date_str)
    if is_demo:
        demo_banner()

    if df.empty:
        st.info("No games for this date.")
        return

    away_col = "away_team" if "away_team" in df.columns else "away_team_id"
    home_col = "home_team" if "home_team" in df.columns else "home_team_id"
    game_options = {
        f"{row[away_col]} @ {row[home_col]}": row["game_id"]
        for _, row in df.iterrows()
        if pd.notna(row.get("game_id"))
    }

    if not game_options:
        st.info("No games available.")
        return

    selected_label = st.selectbox("Select Game", list(game_options.keys()))
    game_id = game_options[selected_label]
    proj_row = df[df["game_id"] == game_id].iloc[0] if game_id in df["game_id"].values else None

    # Metric cards
    if proj_row is not None:
        c1, c2, c3, c4 = st.columns(4)
        mkt  = proj_row.get("market_total")
        mdl  = proj_row.get("ensemble_total")
        diff = proj_row.get("differential")
        conf = proj_row.get("confidence_score")

        c1.metric("Market Total",  f"{mkt:.1f}"  if isinstance(mkt, float) else "—")
        c2.metric("Model Total",   f"{mdl:.1f}"  if isinstance(mdl, float) else "—",
                  delta=f"{diff:+.1f}" if isinstance(diff, float) else None)
        c3.metric("Edge Side",     proj_row.get("edge_side") or "—")
        c4.metric("Confidence",    f"{int(conf*100)}%" if isinstance(conf, float) else "—")

    st.markdown("<br>", unsafe_allow_html=True)

    # Line history chart
    history_df = load_line_history(game_id)

    if history_df.empty:
        st.info("No line history recorded yet. The pipeline logs line snapshots each time it runs.")
        return

    if "timestamp" in history_df.columns:
        history_df["timestamp"] = pd.to_datetime(history_df["timestamp"])

    fig = go.Figure()
    colors = ["#60a5fa", "#f59e0b", "#a78bfa", "#34d399", "#fb7185"]
    books = history_df["sportsbook"].unique() if "sportsbook" in history_df.columns else []

    for i, book in enumerate(books):
        bdf = history_df[history_df["sportsbook"] == book].sort_values("timestamp")
        color = colors[i % len(colors)]
        fig.add_trace(go.Scatter(
            x=bdf["timestamp"],
            y=bdf["total"],
            mode="lines+markers",
            name=book,
            line=dict(color=color, width=2),
            marker=dict(color=color, size=7),
        ))

    if proj_row is not None:
        proj_val = proj_row.get("ensemble_total")
        if isinstance(proj_val, float):
            fig.add_hline(
                y=proj_val,
                line_dash="dash",
                line_color="#f59e0b",
                line_width=1.5,
                annotation_text=f"Model: {proj_val:.1f}",
                annotation_font_color="#f59e0b",
                annotation_font_size=12,
            )

    fig.update_layout(
        paper_bgcolor="#12151e",
        plot_bgcolor="#12151e",
        font=dict(color="#94a3b8", size=12),
        xaxis=dict(gridcolor="#1e2433", title="Time"),
        yaxis=dict(gridcolor="#1e2433", title="Total"),
        legend=dict(bgcolor="#12151e", bordercolor="#1e2433", borderwidth=1),
        height=380,
        margin=dict(l=10, r=30, t=20, b=20),
        title=dict(text=f"Line History — {selected_label}", font=dict(color="#e2e8f0"), x=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Raw table
    with st.expander("Raw line history data"):
        st.dataframe(history_df.sort_values("timestamp", ascending=False), use_container_width=True)


# ── Page: Historical Performance ─────────────────────────────────────────────

def page_historical():
    wh_header("Historical Performance", show_date=False)

    c1, c2 = st.columns(2)
    with c1:
        start = st.date_input("From", value=date.today() - timedelta(days=90))
    with c2:
        end = st.date_input("To",   value=date.today())

    hist_df, is_demo = load_historical()
    if is_demo:
        demo_banner()

    if hist_df.empty:
        st.info("No historical data. Run `python main.py backtest` to generate results.")
        return

    # Filter by date if column exists
    if "game_date" in hist_df.columns:
        hist_df["game_date"] = pd.to_datetime(hist_df["game_date"])
        hist_df = hist_df[
            (hist_df["game_date"] >= pd.Timestamp(start)) &
            (hist_df["game_date"] <= pd.Timestamp(end))
        ]

    if hist_df.empty:
        st.info("No results in selected date range.")
        return

    # ── Top summary metrics ────────────────────────────────────────────────
    results_col = "result" if "result" in hist_df.columns else None

    if results_col:
        wins   = hist_df[results_col].isin(["over", "under"]).sum()
        losses = hist_df[results_col].isin(["lose_over", "lose_under"]).sum()
        pushes = hist_df[results_col].str.lower().eq("push").sum()
        total  = wins + losses
        wp     = f"{wins/total*100:.1f}%" if total > 0 else "—"
        vig    = -110
        roi_units = (wins * (100/110) - losses) / max(total, 1)

        c1, c2, c3, c4, c5 = st.columns(5)
        for col, val, label, css in [
            (c1, str(wins),   "Correct",    "stat-val-green"),
            (c2, str(losses), "Wrong",      "stat-val-red"),
            (c3, str(pushes), "Pushes",     "stat-val-gold"),
            (c4, wp,          "Win Rate",   "stat-val-gold"),
            (c5, f"{roi_units*100:.1f}%", "Flat Bet ROI", "stat-val-green" if roi_units > 0 else "stat-val-red"),
        ]:
            col.markdown(f"""
<div class="stat-card">
  <div class="stat-card-value {css}">{val}</div>
  <div class="stat-card-label">{label}</div>
</div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    # ── Win rate by edge bucket ────────────────────────────────────────────
    if "edge_bucket" in hist_df.columns and results_col:
        section_header("Win Rate by Edge Bucket")
        bucket_stats = []
        for bucket in ["0-2", "2-4", "4-6", "6-8", "8+"]:
            sub = hist_df[hist_df["edge_bucket"] == bucket]
            if len(sub) == 0:
                continue
            w = sub[results_col].isin(["over", "under"]).sum()
            l = sub[results_col].isin(["lose_over", "lose_under"]).sum()
            total_b = w + l
            wp_b = w / total_b * 100 if total_b > 0 else 0
            roi_b = (w * (100/110) - l) / max(total_b, 1) * 100
            bucket_stats.append({"Bucket": bucket, "Bets": total_b, "Win%": round(wp_b, 1), "ROI%": round(roi_b, 1)})

        if bucket_stats:
            bdf = pd.DataFrame(bucket_stats)

            fig = go.Figure()
            bar_colors = [
                "#4ade80" if v >= 52.4 else "#f87171"
                for v in bdf["Win%"]
            ]
            fig.add_trace(go.Bar(
                x=bdf["Bucket"], y=bdf["Win%"],
                marker_color=bar_colors,
                text=[f"{v:.1f}%" for v in bdf["Win%"]],
                textposition="outside",
                textfont=dict(size=13),
            ))
            fig.add_hline(y=52.4, line_dash="dash", line_color="#64748b", line_width=1,
                          annotation_text="Break-even 52.4%", annotation_font_color="#64748b")
            fig.update_layout(
                paper_bgcolor="#12151e", plot_bgcolor="#12151e",
                font=dict(color="#94a3b8", size=12),
                xaxis=dict(gridcolor="#1e2433", title="Edge Bucket (pts)"),
                yaxis=dict(gridcolor="#1e2433", title="Win Rate %", range=[30, 80]),
                height=320, margin=dict(l=10, r=10, t=20, b=20),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                bdf.style.format({"Win%": "{:.1f}%", "ROI%": "{:.1f}%"}),
                use_container_width=True
            )

    # ── Calibration ────────────────────────────────────────────────────────
    if "projected_total" in hist_df.columns and "actual_total" in hist_df.columns:
        section_header("Model Calibration")
        cal = hist_df[["projected_total", "actual_total"]].dropna()
        if not cal.empty:
            errors = cal["projected_total"] - cal["actual_total"]
            mae  = errors.abs().mean()
            rmse = (errors ** 2).mean() ** 0.5

            cc1, cc2, cc3 = st.columns(3)
            cc1.metric("MAE",  f"{mae:.2f} pts")
            cc2.metric("RMSE", f"{rmse:.2f} pts")
            cc3.metric("Bias", f"{errors.mean():+.2f} pts")

            lo = min(cal["actual_total"].min(), cal["projected_total"].min()) - 5
            hi = max(cal["actual_total"].max(), cal["projected_total"].max()) + 5
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=cal["actual_total"], y=cal["projected_total"],
                mode="markers",
                marker=dict(color="#60a5fa", opacity=0.5, size=6),
                name="Games",
            ))
            fig2.add_trace(go.Scatter(
                x=[lo, hi], y=[lo, hi],
                mode="lines",
                name="Perfect",
                line=dict(color="#f59e0b", dash="dash", width=1.5),
            ))
            fig2.update_layout(
                paper_bgcolor="#12151e", plot_bgcolor="#12151e",
                font=dict(color="#94a3b8"),
                xaxis=dict(gridcolor="#1e2433", title="Actual Total"),
                yaxis=dict(gridcolor="#1e2433", title="Projected Total"),
                height=420, margin=dict(l=10, r=10, t=20, b=20),
                legend=dict(bgcolor="#12151e", bordercolor="#1e2433"),
            )
            st.plotly_chart(fig2, use_container_width=True)

    # Detail table
    section_header("Game Log")
    show_cols = [c for c in [
        "game_date", "projected_total", "market_total", "actual_total",
        "differential", "edge_side", "result", "edge_bucket"
    ] if c in hist_df.columns]
    st.dataframe(hist_df[show_cols].head(300), use_container_width=True, height=400)


# ── Page: Team Lookup ─────────────────────────────────────────────────────────

def page_team_lookup():
    wh_header("Team Lookup", show_date=False)

    db = get_db()
    teams = []
    if db:
        try:
            teams = db.get_all_teams() or []
        except Exception:
            pass

    if not teams:
        st.markdown("""
<div style="color:#64748b;font-size:14px;padding:20px 0;">
  No teams in the database yet. Run <code>python main.py run</code> to populate team data.
  <br><br>
  Teams will appear here after the first pipeline run.
</div>""", unsafe_allow_html=True)
        return

    team_options = {t["team_name"]: t["team_id"] for t in teams if t.get("team_name")}
    selected_name = st.selectbox("Select Team", sorted(team_options.keys()))
    selected_id   = team_options[selected_name]
    season        = st.text_input("Season", value=str(date.today().year))

    # Season stats
    stats = {}
    if db:
        try:
            stats = db.get_team_stats(selected_id, season) or {}
        except Exception:
            pass

    if stats:
        section_header("Season Statistics")
        stat_map = {
            "adj_oe": "Adj. Offense (per 100)",
            "adj_de": "Adj. Defense (per 100)",
            "adj_tempo": "Adj. Tempo",
            "ppg": "Points Per Game",
            "opp_ppg": "Opp Points Per Game",
            "efg_pct": "Eff FG%",
            "opp_efg_pct": "Opp Eff FG%",
            "three_p_pct": "3P%",
            "tov_rate": "Turnover Rate",
            "orb_rate": "Off Reb Rate",
        }
        items = [(stat_map.get(k, k.replace("_"," ").title()), v)
                 for k, v in stats.items()
                 if k in stat_map and v is not None]
        cols = st.columns(5)
        for i, (label, val) in enumerate(items):
            with cols[i % 5]:
                try:
                    st.metric(label, f"{float(val):.1f}")
                except (TypeError, ValueError):
                    st.metric(label, str(val))
    else:
        st.info("No stats found for this team/season.")

    # Recent games
    section_header("Recent Games")
    if db:
        try:
            today_str = date.today().strftime("%Y-%m-%d")
            recent = db.get_recent_games(selected_id, 10, today_str) or []
            if recent:
                rdf = pd.DataFrame(recent)
                show = [c for c in ["date","home_team_id","away_team_id","home_score","away_score","total_score","status"] if c in rdf.columns]
                st.dataframe(rdf[show], use_container_width=True, height=300)
            else:
                st.info("No completed game history in database.")
        except Exception as e:
            st.warning(f"Could not load recent games: {e}")


# ── Page: Model Info ──────────────────────────────────────────────────────────

def page_model_info():
    wh_header("Model Info", show_date=False)

    cfg = load_config()
    model_cfg = cfg.get("model", {})
    ens_cfg   = model_cfg.get("ensemble", {})

    # Model file status
    model_file = PROJECT_ROOT / "models" / "cbb_totals_models.joblib"
    if model_file.exists():
        mtime = datetime.fromtimestamp(model_file.stat().st_mtime)
        st.success(f"✓ Trained model loaded — last updated {mtime.strftime('%b %d, %Y %I:%M %p')}")
    else:
        st.warning("⚠ No trained model file found. Run `python main.py train` to train the ML models.")

    section_header("Ensemble Weights")
    c1, c2, c3, c4, c5 = st.columns(5)
    for col, label, val in [
        (c1, "Baseline",      model_cfg.get("baseline_weight", 0.35)),
        (c2, "ML Ensemble",   model_cfg.get("ml_weight", 0.65)),
        (c3, "XGBoost",       ens_cfg.get("xgboost_weight", 0.40)),
        (c4, "LightGBM",      ens_cfg.get("lightgbm_weight", 0.40)),
        (c5, "Ridge",         ens_cfg.get("ridge_weight", 0.20)),
    ]:
        col.markdown(f"""
<div class="stat-card">
  <div class="stat-card-value stat-val-gold">{val:.0%}</div>
  <div class="stat-card-label">{label}</div>
</div>""", unsafe_allow_html=True)

    # League averages
    section_header("League Averages")
    st.json({
        "possessions_per_game": model_cfg.get("league_avg_possessions", 68.5),
        "points_per_100_possessions": model_cfg.get("league_avg_points_per_100", 105.0),
        "home_court_advantage_pts": 3.5,
        "min_games_for_features": model_cfg.get("min_games_for_features", 5),
    })

    # Feature importance
    section_header("Feature Importance (Top 25)")
    fi_df = load_feature_importance()
    if fi_df.empty:
        st.info("No feature importance data — train the model first (`python main.py train`).")
    else:
        top = fi_df.head(25)
        fig = go.Figure(go.Bar(
            x=top["importance"],
            y=top["feature"],
            orientation="h",
            marker_color="#60a5fa",
        ))
        fig.update_layout(
            paper_bgcolor="#12151e", plot_bgcolor="#12151e",
            font=dict(color="#94a3b8", size=12),
            xaxis=dict(gridcolor="#1e2433"),
            yaxis=dict(gridcolor="rgba(0,0,0,0)", categoryorder="total ascending"),
            height=600, margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Raw config
    with st.expander("XGBoost Parameters"):
        st.json(model_cfg.get("xgboost", {}))
    with st.expander("LightGBM Parameters"):
        st.json(model_cfg.get("lightgbm", {}))


# ── Sidebar ───────────────────────────────────────────────────────────────────

def sidebar() -> str:
    # ── Auto-refresh (runs page-wide, every 5 min) ────────────────────────
    refresh_count = st_autorefresh(interval=REFRESH_INTERVAL_MS, key="autorefresh")

    with st.sidebar:
        st.markdown("""
<div style="padding: 12px 0 4px;">
  <span style="font-size:22px;font-weight:900;color:#f59e0b;">Wager</span><span style="font-size:22px;font-weight:900;color:#e2e8f0;">Hub</span>
  <span style="display:block;font-size:10px;color:#475569;letter-spacing:1.2px;margin-top:2px;">CBB TOTALS MODEL</span>
</div>
<div style="height:1px;background:#1e2433;margin:10px 0 16px;"></div>""", unsafe_allow_html=True)

        page = st.radio(
            "nav",
            [
                "🏀  Today's Slate",
                "📈  Best Edges",
                "📉  Line Movement",
                "📋  Historical",
                "🔍  Team Lookup",
                "⚙️  Model Info",
            ],
            label_visibility="collapsed",
        )

        st.markdown('<div style="height:1px;background:#1e2433;margin:16px 0 12px;"></div>', unsafe_allow_html=True)

        # Live status indicator
        now_str = datetime.now().strftime("%I:%M %p")
        st.markdown(f"""
<div style="background:#0e1118;border:1px solid #1e2433;border-radius:8px;padding:10px 12px;margin-bottom:10px;">
  <div style="display:flex;align-items:center;gap:6px;margin-bottom:4px;">
    <span style="width:7px;height:7px;background:#4ade80;border-radius:50%;display:inline-block;"></span>
    <span style="font-size:11px;color:#4ade80;font-weight:600;">LIVE</span>
  </div>
  <div style="font-size:11px;color:#64748b;">Last refresh: {now_str}</div>
  <div style="font-size:10px;color:#374151;margin-top:2px;">Auto-updates every 5 min</div>
</div>""", unsafe_allow_html=True)

        db = get_db()
        if db:
            try:
                count = db.table_row_count("games")
                st.markdown(f'<p style="font-size:11px;color:#475569;">Database: {count} games stored</p>', unsafe_allow_html=True)
            except Exception:
                pass

        if st.button("Refresh Now", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        st.markdown('<div style="height:1px;background:#1e2433;margin:12px 0 10px;"></div>', unsafe_allow_html=True)
        st.markdown('<p style="font-size:10px;color:#1e2433;">WagerHub · Internal Use Only</p>', unsafe_allow_html=True)

    return page


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    page = sidebar()

    if   "Slate"      in page: page_todays_slate()
    elif "Edges"      in page: page_best_edges()
    elif "Movement"   in page: page_line_movement()
    elif "Historical" in page: page_historical()
    elif "Team"       in page: page_team_lookup()
    elif "Model"      in page: page_model_info()


if __name__ == "__main__":
    main()

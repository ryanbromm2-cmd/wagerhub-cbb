"""
Database manager for the CBB Totals Model.
Uses SQLAlchemy 2.x with SQLite (default) or PostgreSQL.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from datetime import datetime, date
from pathlib import Path
from typing import Any, Generator, Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    event,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── ORM Base ──────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


# ── Table Models ──────────────────────────────────────────────────────────────

class Team(Base):
    __tablename__ = "teams"

    team_id = Column(String, primary_key=True)
    team_name = Column(String, nullable=False)
    espn_id = Column(String, nullable=True)
    torvik_id = Column(String, nullable=True)
    conference = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)


class Game(Base):
    __tablename__ = "games"

    game_id = Column(String, primary_key=True)
    date = Column(String, nullable=False, index=True)
    home_team_id = Column(String, nullable=False, index=True)
    away_team_id = Column(String, nullable=False, index=True)
    neutral_site = Column(Boolean, default=False)
    home_score = Column(Float, nullable=True)
    away_score = Column(Float, nullable=True)
    total_score = Column(Float, nullable=True)
    status = Column(String, default="scheduled")          # scheduled|live|final
    espn_game_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class TeamSeasonStats(Base):
    __tablename__ = "team_season_stats"

    id = Column(Integer, primary_key=True, autoincrement=True)
    team_id = Column(String, nullable=False, index=True)
    season = Column(String, nullable=False, index=True)
    games_played = Column(Integer, nullable=True)
    adj_oe = Column(Float, nullable=True)
    adj_de = Column(Float, nullable=True)
    adj_tempo = Column(Float, nullable=True)
    raw_oe = Column(Float, nullable=True)
    raw_de = Column(Float, nullable=True)
    ppg = Column(Float, nullable=True)
    opp_ppg = Column(Float, nullable=True)
    efg_pct = Column(Float, nullable=True)
    opp_efg_pct = Column(Float, nullable=True)
    two_p_pct = Column(Float, nullable=True)
    opp_two_p_pct = Column(Float, nullable=True)
    three_p_pct = Column(Float, nullable=True)
    opp_three_p_pct = Column(Float, nullable=True)
    ft_rate = Column(Float, nullable=True)
    opp_ft_rate = Column(Float, nullable=True)
    tov_rate = Column(Float, nullable=True)
    opp_tov_rate = Column(Float, nullable=True)
    orb_rate = Column(Float, nullable=True)
    drb_rate = Column(Float, nullable=True)
    three_pa_rate = Column(Float, nullable=True)
    opp_three_pa_rate = Column(Float, nullable=True)
    sos = Column(Float, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class GameFeatures(Base):
    __tablename__ = "game_features"

    id = Column(Integer, primary_key=True, autoincrement=True)
    game_id = Column(String, nullable=False, unique=True, index=True)
    feature_json = Column(Text, nullable=False)   # All features as JSON
    created_at = Column(DateTime, default=datetime.utcnow)


class Projection(Base):
    __tablename__ = "projections"

    id = Column(Integer, primary_key=True, autoincrement=True)
    game_id = Column(String, nullable=False, index=True)
    run_timestamp = Column(DateTime, default=datetime.utcnow)
    baseline_total = Column(Float, nullable=True)
    ml_total = Column(Float, nullable=True)
    ensemble_total = Column(Float, nullable=True)
    predicted_home_score = Column(Float, nullable=True)
    predicted_away_score = Column(Float, nullable=True)
    predicted_possessions = Column(Float, nullable=True)
    confidence_score = Column(Float, nullable=True)
    model_version = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class OddsSnapshot(Base):
    __tablename__ = "odds_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    game_id = Column(String, nullable=False, index=True)
    sportsbook = Column(String, nullable=False)
    market_total = Column(Float, nullable=True)
    over_odds = Column(Integer, nullable=True)
    under_odds = Column(Integer, nullable=True)
    snapshot_time = Column(DateTime, default=datetime.utcnow)
    is_opening = Column(Boolean, default=False)
    is_closing = Column(Boolean, default=False)


class BacktestResult(Base):
    __tablename__ = "backtest_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    game_id = Column(String, nullable=False, index=True)
    projected_total = Column(Float, nullable=True)
    market_total = Column(Float, nullable=True)
    actual_total = Column(Float, nullable=True)
    differential = Column(Float, nullable=True)
    edge_side = Column(String, nullable=True)
    result = Column(String, nullable=True)    # over | under | push
    edge_bucket = Column(String, nullable=True)
    season = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class LineHistory(Base):
    __tablename__ = "line_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    game_id = Column(String, nullable=False, index=True)
    sportsbook = Column(String, nullable=False)
    total = Column(Float, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)


# ── Database Manager ──────────────────────────────────────────────────────────

class DatabaseManager:
    """
    Central manager for all database interactions.

    Usage::

        db = DatabaseManager(config)
        db.init_db()
        with db.get_session() as session:
            # use session
    """

    def __init__(self, config: dict):
        self.config = config
        self._engine = None
        self._SessionLocal = None
        self._build_engine()

    # ── Engine construction ────────────────────────────────────────────────

    def _build_engine(self) -> None:
        db_config = self.config.get("database", {})
        db_type = db_config.get("type", "sqlite")

        # Prefer DATABASE_URL env var if set
        database_url = os.environ.get("DATABASE_URL", "")

        if database_url:
            url = database_url
            logger.info("Using DATABASE_URL from environment.")
        elif db_type == "postgres":
            raise ValueError(
                "database.type=postgres but DATABASE_URL env var is not set."
            )
        else:
            # SQLite default
            sqlite_path = db_config.get("sqlite_path", "data/cbb_totals.db")
            # Resolve relative to project root
            project_root = Path(__file__).resolve().parents[2]
            full_path = project_root / sqlite_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            url = f"sqlite:///{full_path}"
            logger.debug(f"Using SQLite database at {full_path}")

        connect_args = {}
        if url.startswith("sqlite"):
            connect_args["check_same_thread"] = False

        self._engine = create_engine(
            url,
            connect_args=connect_args,
            echo=False,
            pool_pre_ping=True,
        )

        # Enable WAL mode for SQLite (better concurrency)
        if url.startswith("sqlite"):
            @event.listens_for(self._engine, "connect")
            def set_sqlite_pragma(dbapi_conn, connection_record):
                cursor = dbapi_conn.cursor()
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()

        self._SessionLocal = sessionmaker(
            bind=self._engine,
            autocommit=False,
            autoflush=False,
            expire_on_commit=False,
        )

    def get_engine(self):
        return self._engine

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Yield a transactional session, committing on success or rolling back on error."""
        session: Session = self._SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # ── Schema ────────────────────────────────────────────────────────────

    def init_db(self) -> None:
        """Create all tables if they don't already exist."""
        Base.metadata.create_all(bind=self._engine)
        logger.info("Database schema initialized.")

    # ── Team methods ──────────────────────────────────────────────────────

    def upsert_team(self, team_data: dict) -> None:
        with self.get_session() as session:
            existing = session.get(Team, team_data["team_id"])
            if existing:
                for k, v in team_data.items():
                    setattr(existing, k, v)
            else:
                session.add(Team(**team_data))

    def get_team(self, team_id: str) -> Optional[dict]:
        with self.get_session() as session:
            row = session.get(Team, team_id)
            if row:
                return {c.name: getattr(row, c.name) for c in Team.__table__.columns}
        return None

    def get_all_teams(self) -> list[dict]:
        with self.get_session() as session:
            rows = session.query(Team).filter(Team.is_active == True).all()
            return [
                {c.name: getattr(r, c.name) for c in Team.__table__.columns}
                for r in rows
            ]

    # ── Game methods ──────────────────────────────────────────────────────

    def upsert_game(self, game_data: dict) -> None:
        with self.get_session() as session:
            existing = session.get(Game, game_data["game_id"])
            if existing:
                game_data["updated_at"] = datetime.utcnow()
                for k, v in game_data.items():
                    setattr(existing, k, v)
            else:
                game_data.setdefault("created_at", datetime.utcnow())
                game_data.setdefault("updated_at", datetime.utcnow())
                session.add(Game(**game_data))

    def get_todays_games(self, game_date: Optional[str] = None) -> list[dict]:
        target = game_date or date.today().strftime("%Y-%m-%d")
        with self.get_session() as session:
            rows = session.query(Game).filter(Game.date == target).all()
            return [
                {c.name: getattr(r, c.name) for c in Game.__table__.columns}
                for r in rows
            ]

    def get_completed_games(self, start_date: str, end_date: str) -> list[dict]:
        with self.get_session() as session:
            rows = (
                session.query(Game)
                .filter(
                    Game.date >= start_date,
                    Game.date <= end_date,
                    Game.status == "final",
                    Game.total_score.isnot(None),
                )
                .order_by(Game.date)
                .all()
            )
            return [
                {c.name: getattr(r, c.name) for c in Game.__table__.columns}
                for r in rows
            ]

    def get_game(self, game_id: str) -> Optional[dict]:
        with self.get_session() as session:
            row = session.get(Game, game_id)
            if row:
                return {c.name: getattr(row, c.name) for c in Game.__table__.columns}
        return None

    # ── Team stats methods ────────────────────────────────────────────────

    def upsert_team_stats(self, stats_data: dict) -> None:
        with self.get_session() as session:
            existing = (
                session.query(TeamSeasonStats)
                .filter(
                    TeamSeasonStats.team_id == stats_data["team_id"],
                    TeamSeasonStats.season == stats_data["season"],
                )
                .first()
            )
            if existing:
                stats_data["updated_at"] = datetime.utcnow()
                for k, v in stats_data.items():
                    if hasattr(existing, k):
                        setattr(existing, k, v)
            else:
                stats_data.setdefault("updated_at", datetime.utcnow())
                session.add(TeamSeasonStats(**stats_data))

    def get_team_stats(self, team_id: str, season: str) -> Optional[dict]:
        with self.get_session() as session:
            row = (
                session.query(TeamSeasonStats)
                .filter(
                    TeamSeasonStats.team_id == team_id,
                    TeamSeasonStats.season == season,
                )
                .first()
            )
            if row:
                return {
                    c.name: getattr(row, c.name)
                    for c in TeamSeasonStats.__table__.columns
                }
        return None

    # ── Recent games ──────────────────────────────────────────────────────

    def get_recent_games(
        self,
        team_id: str,
        n: int,
        before_date: str,
    ) -> list[dict]:
        """
        Return the last N completed games for a team before before_date.
        Team can be either home or away.
        """
        with self.get_session() as session:
            rows = (
                session.query(Game)
                .filter(
                    ((Game.home_team_id == team_id) | (Game.away_team_id == team_id)),
                    Game.date < before_date,
                    Game.status == "final",
                    Game.total_score.isnot(None),
                )
                .order_by(Game.date.desc())
                .limit(n)
                .all()
            )
            return [
                {c.name: getattr(r, c.name) for c in Game.__table__.columns}
                for r in rows
            ]

    # ── Feature methods ───────────────────────────────────────────────────

    def save_game_features(self, game_id: str, features: dict) -> None:
        with self.get_session() as session:
            existing = (
                session.query(GameFeatures)
                .filter(GameFeatures.game_id == game_id)
                .first()
            )
            feature_json = json.dumps(features, default=str)
            if existing:
                existing.feature_json = feature_json
            else:
                session.add(
                    GameFeatures(game_id=game_id, feature_json=feature_json)
                )

    def get_game_features(self, game_id: str) -> Optional[dict]:
        with self.get_session() as session:
            row = (
                session.query(GameFeatures)
                .filter(GameFeatures.game_id == game_id)
                .first()
            )
            if row:
                return json.loads(row.feature_json)
        return None

    # ── Projection methods ────────────────────────────────────────────────

    def save_projection(self, projection_data: dict) -> None:
        with self.get_session() as session:
            projection_data.setdefault("run_timestamp", datetime.utcnow())
            projection_data.setdefault("created_at", datetime.utcnow())
            session.add(Projection(**projection_data))

    def get_latest_projection(self, game_id: str) -> Optional[dict]:
        with self.get_session() as session:
            row = (
                session.query(Projection)
                .filter(Projection.game_id == game_id)
                .order_by(Projection.run_timestamp.desc())
                .first()
            )
            if row:
                return {
                    c.name: getattr(row, c.name)
                    for c in Projection.__table__.columns
                }
        return None

    def get_historical_projections(
        self, start_date: str, end_date: str
    ) -> list[dict]:
        """Join projections with games to filter by game date."""
        with self.get_session() as session:
            rows = (
                session.query(Projection, Game)
                .join(Game, Projection.game_id == Game.game_id)
                .filter(Game.date >= start_date, Game.date <= end_date)
                .order_by(Game.date.desc(), Projection.run_timestamp.desc())
                .all()
            )
            results = []
            for proj, game in rows:
                d = {
                    c.name: getattr(proj, c.name)
                    for c in Projection.__table__.columns
                }
                d["game_date"] = game.date
                d["home_team_id"] = game.home_team_id
                d["away_team_id"] = game.away_team_id
                d["actual_total"] = game.total_score
                results.append(d)
            return results

    def get_todays_projections(self, game_date: Optional[str] = None) -> list[dict]:
        target = game_date or date.today().strftime("%Y-%m-%d")
        return self.get_historical_projections(target, target)

    # ── Odds methods ──────────────────────────────────────────────────────

    def save_odds_snapshot(self, odds_data: dict) -> None:
        with self.get_session() as session:
            odds_data.setdefault("snapshot_time", datetime.utcnow())
            session.add(OddsSnapshot(**odds_data))

    def get_latest_odds(self, game_id: str, sportsbook: str = "consensus") -> Optional[dict]:
        with self.get_session() as session:
            row = (
                session.query(OddsSnapshot)
                .filter(
                    OddsSnapshot.game_id == game_id,
                    OddsSnapshot.sportsbook == sportsbook,
                )
                .order_by(OddsSnapshot.snapshot_time.desc())
                .first()
            )
            if row:
                return {
                    c.name: getattr(row, c.name)
                    for c in OddsSnapshot.__table__.columns
                }
        return None

    def get_odds_for_date(self, game_date: str) -> list[dict]:
        """Get most recent odds snapshot per game for a given date."""
        with self.get_session() as session:
            # Subquery: get max snapshot_time per game_id
            subq = (
                session.query(
                    OddsSnapshot.game_id,
                    OddsSnapshot.sportsbook,
                    text("MAX(snapshot_time) as max_time"),
                )
                .join(Game, OddsSnapshot.game_id == Game.game_id)
                .filter(Game.date == game_date)
                .group_by(OddsSnapshot.game_id, OddsSnapshot.sportsbook)
                .subquery()
            )
            rows = (
                session.query(OddsSnapshot)
                .join(
                    subq,
                    (OddsSnapshot.game_id == subq.c.game_id)
                    & (OddsSnapshot.sportsbook == subq.c.sportsbook)
                    & (OddsSnapshot.snapshot_time == subq.c.max_time),
                )
                .all()
            )
            return [
                {c.name: getattr(r, c.name) for c in OddsSnapshot.__table__.columns}
                for r in rows
            ]

    # ── Line history ──────────────────────────────────────────────────────

    def save_line_history(self, entry: dict) -> None:
        with self.get_session() as session:
            entry.setdefault("timestamp", datetime.utcnow())
            session.add(LineHistory(**entry))

    def get_line_history(self, game_id: str, sportsbook: str = None) -> list[dict]:
        with self.get_session() as session:
            q = session.query(LineHistory).filter(LineHistory.game_id == game_id)
            if sportsbook:
                q = q.filter(LineHistory.sportsbook == sportsbook)
            rows = q.order_by(LineHistory.timestamp).all()
            return [
                {c.name: getattr(r, c.name) for c in LineHistory.__table__.columns}
                for r in rows
            ]

    # ── Backtest results ──────────────────────────────────────────────────

    def save_backtest_result(self, result_data: dict) -> None:
        with self.get_session() as session:
            result_data.setdefault("created_at", datetime.utcnow())
            session.add(BacktestResult(**result_data))

    def get_backtest_results(
        self, start_date: str = None, end_date: str = None
    ) -> list[dict]:
        with self.get_session() as session:
            q = session.query(BacktestResult)
            if start_date or end_date:
                q = q.join(Game, BacktestResult.game_id == Game.game_id)
                if start_date:
                    q = q.filter(Game.date >= start_date)
                if end_date:
                    q = q.filter(Game.date <= end_date)
            rows = q.order_by(BacktestResult.created_at.desc()).all()
            return [
                {c.name: getattr(r, c.name) for c in BacktestResult.__table__.columns}
                for r in rows
            ]

    # ── Utility ───────────────────────────────────────────────────────────

    def execute_raw(self, sql: str, params: dict = None) -> list[dict]:
        """Execute raw SQL and return list of row dicts."""
        with self._engine.connect() as conn:
            result = conn.execute(text(sql), params or {})
            cols = list(result.keys())
            return [dict(zip(cols, row)) for row in result.fetchall()]

    def table_row_count(self, table_name: str) -> int:
        rows = self.execute_raw(f"SELECT COUNT(*) as cnt FROM {table_name}")
        return rows[0]["cnt"] if rows else 0

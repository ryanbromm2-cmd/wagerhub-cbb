"""
APScheduler-based scheduler daemon for the CBB Totals Model.
Runs the full pipeline on a daily schedule with intraday odds refreshes.
"""

from __future__ import annotations

import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

import yaml

try:
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR, EVENT_JOB_MISSED
    APSCHEDULER_AVAILABLE = True
except ImportError:
    APSCHEDULER_AVAILABLE = False

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Job state tracking ────────────────────────────────────────────────────────
_job_stats: dict = {
    "morning_run": {"runs": 0, "errors": 0, "last_run": None, "last_duration": None},
    "midday_refresh": {"runs": 0, "errors": 0, "last_run": None, "last_duration": None},
    "evening_refresh": {"runs": 0, "errors": 0, "last_run": None, "last_duration": None},
}


def _load_config() -> dict:
    cfg_path = PROJECT_ROOT / "config" / "config.yaml"
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def _get_db(config: dict):
    from src.utils.db import DatabaseManager
    db = DatabaseManager(config)
    db.init_db()
    return db


# ── Job functions ─────────────────────────────────────────────────────────────

def job_morning_run():
    """
    Full pipeline job: schedule + stats + ML projections + odds + output.
    Runs at the configured morning time (default 9:00 AM ET).
    """
    job_name = "morning_run"
    start_time = time.time()
    logger.info(f"[SCHEDULER] Starting {job_name}...")
    _job_stats[job_name]["runs"] += 1
    _job_stats[job_name]["last_run"] = datetime.now().isoformat()

    try:
        config = _load_config()
        db = _get_db(config)

        from src.pipeline.daily_pipeline import DailyPipeline
        pipeline = DailyPipeline(config, db)
        result = pipeline.run()

        duration = round(time.time() - start_time, 2)
        _job_stats[job_name]["last_duration"] = duration

        logger.info(
            f"[SCHEDULER] {job_name} complete in {duration}s. "
            f"Games: {result.get('games_processed', 0)}, "
            f"With odds: {result.get('games_with_odds', 0)}"
        )

        # Notify on success
        _send_notification(
            config,
            f"Morning Run Complete",
            f"Processed {result.get('games_processed', 0)} games in {duration}s.",
            "info"
        )

    except Exception as exc:
        _job_stats[job_name]["errors"] += 1
        duration = round(time.time() - start_time, 2)
        logger.error(f"[SCHEDULER] {job_name} FAILED after {duration}s: {exc}", exc_info=True)
        _schedule_retry(job_name, job_morning_run, delay_minutes=5)


def job_midday_refresh():
    """
    Intraday odds refresh job.
    Re-pulls current odds and recomputes edges.
    Runs at the configured midday time (default 12:00 PM ET).
    """
    job_name = "midday_refresh"
    start_time = time.time()
    logger.info(f"[SCHEDULER] Starting {job_name}...")
    _job_stats[job_name]["runs"] += 1
    _job_stats[job_name]["last_run"] = datetime.now().isoformat()

    try:
        config = _load_config()
        db = _get_db(config)

        from src.pipeline.daily_pipeline import DailyPipeline
        pipeline = DailyPipeline(config, db)
        result = pipeline.refresh_odds_only()

        duration = round(time.time() - start_time, 2)
        _job_stats[job_name]["last_duration"] = duration
        logger.info(
            f"[SCHEDULER] {job_name} complete in {duration}s. "
            f"Games updated: {result.get('updated', 0)}"
        )

    except Exception as exc:
        _job_stats[job_name]["errors"] += 1
        duration = round(time.time() - start_time, 2)
        logger.error(f"[SCHEDULER] {job_name} FAILED after {duration}s: {exc}", exc_info=True)
        _schedule_retry(job_name, job_midday_refresh, delay_minutes=5)


def job_evening_refresh():
    """
    Evening odds refresh job.
    Runs at the configured evening time (default 5:00 PM ET).
    Also triggers alert check on final lines.
    """
    job_name = "evening_refresh"
    start_time = time.time()
    logger.info(f"[SCHEDULER] Starting {job_name}...")
    _job_stats[job_name]["runs"] += 1
    _job_stats[job_name]["last_run"] = datetime.now().isoformat()

    try:
        config = _load_config()
        db = _get_db(config)

        from src.pipeline.daily_pipeline import DailyPipeline
        pipeline = DailyPipeline(config, db)
        result = pipeline.refresh_odds_only()

        duration = round(time.time() - start_time, 2)
        _job_stats[job_name]["last_duration"] = duration
        logger.info(
            f"[SCHEDULER] {job_name} complete in {duration}s. "
            f"Games updated: {result.get('updated', 0)}"
        )

    except Exception as exc:
        _job_stats[job_name]["errors"] += 1
        duration = round(time.time() - start_time, 2)
        logger.error(f"[SCHEDULER] {job_name} FAILED after {duration}s: {exc}", exc_info=True)
        _schedule_retry(job_name, job_evening_refresh, delay_minutes=5)


# ── Retry helper ──────────────────────────────────────────────────────────────

_retry_scheduler: "BackgroundScheduler | None" = None


def _schedule_retry(job_name: str, job_fn, delay_minutes: int = 5) -> None:
    """Schedule a one-time retry of the given job after delay_minutes."""
    global _retry_scheduler
    if not APSCHEDULER_AVAILABLE:
        return

    logger.info(f"[SCHEDULER] Scheduling retry for {job_name} in {delay_minutes} minutes.")
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.date import DateTrigger
    from datetime import datetime, timedelta

    if _retry_scheduler is None or not _retry_scheduler.running:
        _retry_scheduler = BackgroundScheduler()
        _retry_scheduler.start()

    run_at = datetime.now() + timedelta(minutes=delay_minutes)
    _retry_scheduler.add_job(
        job_fn,
        trigger=DateTrigger(run_date=run_at),
        id=f"retry_{job_name}_{run_at.strftime('%H%M%S')}",
        replace_existing=True,
        name=f"Retry {job_name}",
    )


# ── Event listeners ───────────────────────────────────────────────────────────

def on_job_executed(event):
    logger.debug(f"[SCHEDULER] Job {event.job_id} executed successfully.")


def on_job_error(event):
    logger.error(
        f"[SCHEDULER] Job {event.job_id} raised an exception: {event.exception}"
    )


def on_job_missed(event):
    logger.warning(f"[SCHEDULER] Job {event.job_id} was missed (scheduled time passed).")


# ── Notification helper ───────────────────────────────────────────────────────

def _send_notification(config: dict, title: str, body: str, level: str = "info") -> None:
    """Send a notification via the AlertManager if configured."""
    try:
        from src.utils.alerts import AlertManager
        am = AlertManager(config)
        am.notify(title, body, level)
    except Exception:
        pass


# ── Scheduler startup / shutdown ──────────────────────────────────────────────

_main_scheduler: "BlockingScheduler | None" = None


def start_scheduler(daemon: bool = False) -> None:
    """
    Start the APScheduler.

    Args:
        daemon: If True, run as a background scheduler (non-blocking).
                If False (default), run as a blocking scheduler.
    """
    global _main_scheduler

    if not APSCHEDULER_AVAILABLE:
        logger.error(
            "APScheduler is not installed. "
            "Install it with: pip install APScheduler>=3.10.0"
        )
        sys.exit(1)

    config = _load_config()
    sched_cfg = config.get("scheduler", {})
    timezone = sched_cfg.get("timezone", "America/New_York")

    # Parse run times
    morning_time = sched_cfg.get("morning_run_time", "09:00")
    midday_time = sched_cfg.get("midday_run_time", "12:00")
    evening_time = sched_cfg.get("evening_run_time", "17:00")

    morning_h, morning_m = _parse_time(morning_time)
    midday_h, midday_m = _parse_time(midday_time)
    evening_h, evening_m = _parse_time(evening_time)

    # Choose scheduler type
    if daemon:
        from apscheduler.schedulers.background import BackgroundScheduler as Sched
    else:
        from apscheduler.schedulers.blocking import BlockingScheduler as Sched

    _main_scheduler = Sched(timezone=timezone)

    # Register event listeners
    _main_scheduler.add_listener(on_job_executed, EVENT_JOB_EXECUTED)
    _main_scheduler.add_listener(on_job_error, EVENT_JOB_ERROR)
    _main_scheduler.add_listener(on_job_missed, EVENT_JOB_MISSED)

    # Schedule jobs
    _main_scheduler.add_job(
        job_morning_run,
        trigger=CronTrigger(hour=morning_h, minute=morning_m, timezone=timezone),
        id="morning_run",
        name="Morning Run (full pipeline)",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=600,   # Allow 10-min late start
    )

    _main_scheduler.add_job(
        job_midday_refresh,
        trigger=CronTrigger(hour=midday_h, minute=midday_m, timezone=timezone),
        id="midday_refresh",
        name="Midday Odds Refresh",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=300,
    )

    _main_scheduler.add_job(
        job_evening_refresh,
        trigger=CronTrigger(hour=evening_h, minute=evening_m, timezone=timezone),
        id="evening_refresh",
        name="Evening Odds Refresh",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=300,
    )

    logger.info(
        f"[SCHEDULER] Starting with timezone={timezone}\n"
        f"  Morning run:   {morning_h:02d}:{morning_m:02d}\n"
        f"  Midday refresh: {midday_h:02d}:{midday_m:02d}\n"
        f"  Evening refresh: {evening_h:02d}:{evening_m:02d}"
    )

    # Graceful shutdown handler
    def handle_signal(signum, frame):
        logger.info("[SCHEDULER] Received shutdown signal.")
        stop_scheduler()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    print(f"\nScheduler running (Ctrl+C to stop).")
    print(f"Jobs: morning={morning_time}, midday={midday_time}, evening={evening_time} [{timezone}]")
    print(f"Logs: {PROJECT_ROOT / 'logs' / 'cbb_totals.log'}\n")

    if daemon:
        _main_scheduler.start()
        # Non-blocking: keep alive
        try:
            while True:
                time.sleep(60)
        except (KeyboardInterrupt, SystemExit):
            stop_scheduler()
    else:
        # Blocking — runs until killed
        try:
            _main_scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            stop_scheduler()


def stop_scheduler() -> None:
    """Gracefully stop the scheduler."""
    global _main_scheduler, _retry_scheduler

    if _main_scheduler and _main_scheduler.running:
        logger.info("[SCHEDULER] Shutting down main scheduler...")
        _main_scheduler.shutdown(wait=False)
        logger.info("[SCHEDULER] Main scheduler stopped.")

    if _retry_scheduler and _retry_scheduler.running:
        _retry_scheduler.shutdown(wait=False)


def get_scheduler_status() -> dict:
    """Return current scheduler status and job statistics."""
    jobs = []
    if _main_scheduler and _main_scheduler.running:
        for job in _main_scheduler.get_jobs():
            jobs.append(
                {
                    "id": job.id,
                    "name": job.name,
                    "next_run": str(job.next_run_time) if job.next_run_time else None,
                }
            )

    return {
        "running": _main_scheduler.running if _main_scheduler else False,
        "jobs": jobs,
        "stats": _job_stats,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_time(time_str: str) -> tuple[int, int]:
    """Parse 'HH:MM' string into (hour, minute) ints."""
    try:
        parts = time_str.strip().split(":")
        return int(parts[0]), int(parts[1])
    except (IndexError, ValueError):
        logger.warning(f"Invalid time format '{time_str}', defaulting to 09:00.")
        return 9, 0


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    start_scheduler()

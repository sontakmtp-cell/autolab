"""State management, SQLite database queries, and time utilities for PAXG Forecast Lab Web UI."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import time
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from ..constants import TIMEFRAME_1H, TIMEFRAME_4H, get_horizon_for_timeframe
from ..queue.storage import GPUJobStorage
from ..queue.types import AutoRunState, JobPriority, JobSpec, JobStatus, JobType

DEFAULT_DB_PATH = Path("var/paxg_lab/paxg_lab.db")
VIETNAM_TZ = ZoneInfo("Asia/Ho_Chi_Minh")


def timestamp_to_vietnam_str(ts_ms: int | float | None, fmt: str = "%d/%m/%Y %H:%M") -> str:
    """Converts a millisecond or second timestamp to formatted Vietnam time (UTC+7)."""
    if ts_ms is None:
        return "N/A"
    # Detect whether ts is in milliseconds or seconds
    sec = ts_ms / 1000.0 if ts_ms > 1e11 else float(ts_ms)
    dt = datetime.fromtimestamp(sec, tz=timezone.utc).astimezone(VIETNAM_TZ)
    return dt.strftime(fmt)


def get_db_connection(db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Returns a SQLite connection with WAL mode and timeout."""
    conn = sqlite3.connect(str(db_path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=30000;")
    return conn


def get_latest_candle(timeframe: str = TIMEFRAME_1H, db_path: Path = DEFAULT_DB_PATH) -> dict[str, Any] | None:
    """Retrieves the latest closed candle from SQLite klines table."""
    tf = timeframe.lower().strip()
    if not db_path.exists():
        return None
    try:
        with get_db_connection(db_path) as conn:
            cur = conn.execute(
                """
                SELECT open_time, close_time, open, high, low, close, volume, quote_volume, count
                FROM klines
                WHERE timeframe = ?
                ORDER BY open_time DESC
                LIMIT 1;
                """,
                (tf,),
            )
            row = cur.fetchone()
            if row:
                return dict(row)
    except Exception:
        pass
    return None


def get_candle_count(timeframe: str = TIMEFRAME_1H, db_path: Path = DEFAULT_DB_PATH) -> int:
    """Returns the total number of closed candles available for a timeframe."""
    tf = timeframe.lower().strip()
    if not db_path.exists():
        return 0
    try:
        with get_db_connection(db_path) as conn:
            cur = conn.execute("SELECT COUNT(*) AS total FROM klines WHERE timeframe = ?;", (tf,))
            row = cur.fetchone()
            return int(row["total"]) if row else 0
    except Exception:
        return 0


def get_recent_candles(
    timeframe: str = TIMEFRAME_1H,
    limit: int = 100,
    db_path: Path = DEFAULT_DB_PATH,
) -> pd.DataFrame:
    """Fetches the most recent candles for Plotly candlestick visualization."""
    tf = timeframe.lower().strip()
    if not db_path.exists():
        return pd.DataFrame()
    try:
        with get_db_connection(db_path) as conn:
            cur = conn.execute(
                """
                SELECT open_time, open, high, low, close, volume, quote_volume
                FROM klines
                WHERE timeframe = ?
                ORDER BY open_time DESC
                LIMIT ?;
                """,
                (tf, limit),
            )
            rows = cur.fetchall()
            if not rows:
                return pd.DataFrame()
            data = [dict(r) for r in reversed(rows)]
            df = pd.DataFrame(data)
            # Add Vietnam time column for clean plotting
            df["time_vn"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_convert(VIETNAM_TZ)
            return df
    except Exception:
        return pd.DataFrame()


def get_gpu_queue_summary(db_path: Path = DEFAULT_DB_PATH) -> dict[str, Any]:
    """Summarizes current GPU queue state: running job, queued count, auto-run states."""
    storage = GPUJobStorage(db_path)
    running_job = storage.get_running_job()
    
    queued_jobs = storage.list_jobs(status=JobStatus.QUEUED.value, limit=100)
    queued_count = len(queued_jobs)

    state_1h = storage.get_auto_run_state("1h")
    state_4h = storage.get_auto_run_state("4h")

    lease = storage.get_coordinator_lease()
    is_scheduler_alive = lease is not None and (time.time() - float(lease.get("heartbeat", 0))) < 35.0

    return {
        "running_job": running_job,
        "queued_count": queued_count,
        "auto_state_1h": state_1h,
        "auto_state_4h": state_4h,
        "scheduler_alive": is_scheduler_alive,
        "coordinator_pid": lease.get("pid") if lease else None,
    }


def get_latest_snapshot_path(timeframe: str = TIMEFRAME_1H, snapshots_dir: Path = Path("var/paxg_lab/snapshots")) -> Path | None:
    """Finds the most recent valid snapshot directory or file for the timeframe."""
    if not snapshots_dir.exists():
        return None
    tf = timeframe.lower().strip()
    matches = sorted(
        [
            p for p in snapshots_dir.glob(f"paxgusdt_{tf}_*")
            if (p.is_dir() and (p / "data.npz").exists()) or (p.is_file() and p.suffix == ".npz")
        ],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return matches[0] if matches else None


def list_recent_jobs(
    job_type: str | None = None,
    timeframe: str | None = None,
    limit: int = 15,
    db_path: Path = DEFAULT_DB_PATH,
) -> list[JobSpec]:
    """Lists recent jobs filtered by type and timeframe."""
    storage = GPUJobStorage(db_path)
    return storage.list_jobs(job_type=job_type, timeframe=timeframe, limit=limit)

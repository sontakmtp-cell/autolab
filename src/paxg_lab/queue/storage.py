"""SQLite persistent storage and transaction manager for GPU Queue in PAXG Forecast Lab."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import time
from typing import Any

from .types import AutoRunState, JobPriority, JobSpec, JobStatus

DEFAULT_DB_PATH = Path("var/paxg_lab/paxg_lab.db")


class GPUJobStorage:
    """Manages SQLite job queue persistence, concurrency locks, and state machines."""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def get_connection(self) -> sqlite3.Connection:
        """Returns SQLite connection with WAL mode and row factory."""
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=30000;")
        return conn

    def init_db(self) -> None:
        """Initializes tables and indexes for GPU job coordination."""
        with self.get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS gpu_jobs (
                    job_id TEXT PRIMARY KEY,
                    job_type TEXT NOT NULL,
                    timeframe TEXT,
                    priority INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    result TEXT,
                    error_message TEXT,
                    idempotency_key TEXT,
                    worker_pid INTEGER,
                    worker_create_time REAL,
                    timeout_seconds REAL NOT NULL DEFAULT 1200.0,
                    created_at REAL NOT NULL,
                    started_at REAL,
                    finished_at REAL,
                    heartbeat_at REAL,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    progress_pct REAL NOT NULL DEFAULT 0.0,
                    progress_message TEXT NOT NULL DEFAULT ''
                );

                CREATE INDEX IF NOT EXISTS idx_gpu_jobs_status_priority
                ON gpu_jobs (status, priority, created_at);

                CREATE INDEX IF NOT EXISTS idx_gpu_jobs_idempotency
                ON gpu_jobs (idempotency_key);

                CREATE TABLE IF NOT EXISTS scheduler_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );
            """)

    def submit_job(self, job_spec: JobSpec) -> str:
        """Submits a job to the queue, enforcing idempotency deduplication on active jobs."""
        now = time.time()
        payload_json = json.dumps(job_spec.payload, default=str)
        result_json = json.dumps(job_spec.result, default=str) if job_spec.result is not None else None

        with self.get_connection() as conn:
            # Check idempotency: if job with same key is QUEUED or RUNNING, return existing job_id
            if job_spec.idempotency_key:
                cur = conn.execute(
                    """
                    SELECT job_id FROM gpu_jobs
                    WHERE idempotency_key = ? AND status IN ('QUEUED', 'RUNNING')
                    ORDER BY created_at DESC LIMIT 1;
                    """,
                    (job_spec.idempotency_key,),
                )
                row = cur.fetchone()
                if row:
                    return str(row["job_id"])

            conn.execute(
                """
                INSERT INTO gpu_jobs (
                    job_id, job_type, timeframe, priority, status,
                    payload, result, error_message, idempotency_key,
                    worker_pid, worker_create_time, timeout_seconds,
                    created_at, started_at, finished_at, heartbeat_at,
                    cancel_requested, progress_pct, progress_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    job_spec.job_id,
                    job_spec.job_type,
                    job_spec.timeframe,
                    job_spec.priority,
                    JobStatus.QUEUED.value,
                    payload_json,
                    result_json,
                    job_spec.error_message,
                    job_spec.idempotency_key,
                    job_spec.worker_pid,
                    job_spec.worker_create_time,
                    job_spec.timeout_seconds,
                    job_spec.created_at or now,
                    job_spec.started_at,
                    job_spec.finished_at,
                    job_spec.heartbeat_at,
                    1 if job_spec.cancel_requested else 0,
                    job_spec.progress_pct,
                    job_spec.progress_message,
                ),
            )
            return job_spec.job_id

    def get_job(self, job_id: str) -> JobSpec | None:
        """Loads a JobSpec by its job_id."""
        with self.get_connection() as conn:
            cur = conn.execute("SELECT * FROM gpu_jobs WHERE job_id = ?;", (job_id,))
            row = cur.fetchone()
            if not row:
                return None
            return JobSpec.from_row(dict(row))

    def get_running_job(self) -> JobSpec | None:
        """Returns the currently RUNNING job, if any (enforcing at most one)."""
        with self.get_connection() as conn:
            cur = conn.execute(
                "SELECT * FROM gpu_jobs WHERE status = ? ORDER BY started_at ASC LIMIT 1;",
                (JobStatus.RUNNING.value,),
            )
            row = cur.fetchone()
            if not row:
                return None
            return JobSpec.from_row(dict(row))

    def list_jobs(
        self,
        status: str | None = None,
        job_type: str | None = None,
        timeframe: str | None = None,
        limit: int = 50,
    ) -> list[JobSpec]:
        """Lists jobs matching filters ordered by created_at DESC."""
        query = ["SELECT * FROM gpu_jobs WHERE 1=1"]
        params: list[Any] = []

        if status is not None:
            query.append("AND status = ?")
            params.append(status)
        if job_type is not None:
            query.append("AND job_type = ?")
            params.append(job_type)
        if timeframe is not None:
            query.append("AND timeframe = ?")
            params.append(timeframe)

        query.append("ORDER BY created_at DESC LIMIT ?;")
        params.append(limit)

        with self.get_connection() as conn:
            cur = conn.execute(" ".join(query), params)
            rows = cur.fetchall()
            return [JobSpec.from_row(dict(r)) for r in rows]

    def acquire_next_job(self, last_auto_timeframe: str | None = None) -> JobSpec | None:
        """Atomically selects and marks the next eligible job as RUNNING.
        
        Enforces:
        1. Single GPU Worker invariant: if any job is currently RUNNING, returns None immediately.
        2. Priority hierarchy: Priority 1 (FORECAST) > Priority 2 (MANUAL) > Priority 3 (AUTO).
        3. 1h / 4h Alternating Round-Robin for Priority 3 (AUTO) jobs.
        """
        now = time.time()
        with self.get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE;")

            # 1. Enforce single GPU process invariant
            cur = conn.execute(
                "SELECT job_id FROM gpu_jobs WHERE status = ? LIMIT 1;",
                (JobStatus.RUNNING.value,),
            )
            if cur.fetchone():
                conn.commit()
                return None  # A job is already active on the GPU

            # 2. Priority 1: Instant Forecast (FIFO)
            cur = conn.execute(
                """
                SELECT * FROM gpu_jobs
                WHERE status = ? AND priority = ?
                ORDER BY created_at ASC LIMIT 1;
                """,
                (JobStatus.QUEUED.value, JobPriority.FORECAST.value),
            )
            candidate = cur.fetchone()

            # 3. Priority 2: Manual Work (FIFO)
            if not candidate:
                cur = conn.execute(
                    """
                    SELECT * FROM gpu_jobs
                    WHERE status = ? AND priority = ?
                    ORDER BY created_at ASC LIMIT 1;
                    """,
                    (JobStatus.QUEUED.value, JobPriority.MANUAL.value),
                )
                candidate = cur.fetchone()

            # 4. Priority 3: Autonomous Tuning with 1h/4h Round-Robin Alternation
            if not candidate:
                if last_auto_timeframe == "1h":
                    # Prefer 4h next
                    cur = conn.execute(
                        """
                        SELECT * FROM gpu_jobs
                        WHERE status = ? AND priority = ? AND timeframe = '4h'
                        ORDER BY created_at ASC LIMIT 1;
                        """,
                        (JobStatus.QUEUED.value, JobPriority.AUTO.value),
                    )
                    candidate = cur.fetchone()
                elif last_auto_timeframe == "4h":
                    # Prefer 1h next
                    cur = conn.execute(
                        """
                        SELECT * FROM gpu_jobs
                        WHERE status = ? AND priority = ? AND timeframe = '1h'
                        ORDER BY created_at ASC LIMIT 1;
                        """,
                        (JobStatus.QUEUED.value, JobPriority.AUTO.value),
                    )
                    candidate = cur.fetchone()

                # Fallback to any remaining auto job (FIFO)
                if not candidate:
                    cur = conn.execute(
                        """
                        SELECT * FROM gpu_jobs
                        WHERE status = ? AND priority = ?
                        ORDER BY created_at ASC LIMIT 1;
                        """,
                        (JobStatus.QUEUED.value, JobPriority.AUTO.value),
                    )
                    candidate = cur.fetchone()

            if not candidate:
                conn.commit()
                return None

            selected_id = candidate["job_id"]
            conn.execute(
                """
                UPDATE gpu_jobs
                SET status = ?, started_at = ?, heartbeat_at = ?
                WHERE job_id = ?;
                """,
                (JobStatus.RUNNING.value, now, now, selected_id),
            )
            conn.commit()

            cur = conn.execute("SELECT * FROM gpu_jobs WHERE job_id = ?;", (selected_id,))
            updated_row = cur.fetchone()
            return JobSpec.from_row(dict(updated_row))

    def register_worker(self, job_id: str, pid: int, create_time: float) -> None:
        """Registers the subprocess PID and process creation timestamp to avoid PID recycling."""
        now = time.time()
        with self.get_connection() as conn:
            conn.execute(
                """
                UPDATE gpu_jobs
                SET worker_pid = ?, worker_create_time = ?, heartbeat_at = ?
                WHERE job_id = ?;
                """,
                (pid, create_time, now, job_id),
            )

    def update_heartbeat(
        self,
        job_id: str,
        progress_pct: float | None = None,
        progress_message: str | None = None,
    ) -> bool:
        """Updates worker heartbeat and progress. Returns True if cancellation was requested."""
        now = time.time()
        with self.get_connection() as conn:
            updates = ["heartbeat_at = ?"]
            params: list[Any] = [now]

            if progress_pct is not None:
                updates.append("progress_pct = ?")
                params.append(min(max(progress_pct, 0.0), 100.0))
            if progress_message is not None:
                updates.append("progress_message = ?")
                params.append(progress_message)

            params.append(job_id)
            conn.execute(f"UPDATE gpu_jobs SET {', '.join(updates)} WHERE job_id = ?;", params)

            cur = conn.execute("SELECT cancel_requested FROM gpu_jobs WHERE job_id = ?;", (job_id,))
            row = cur.fetchone()
            if row and row["cancel_requested"]:
                return True
            return False

    def request_cancel(self, job_id: str) -> bool:
        """Requests cancellation of a job.
        
        If QUEUED: transitions immediately to CANCELLED.
        If RUNNING: sets cancel_requested flag so worker can halt gracefully.
        """
        now = time.time()
        with self.get_connection() as conn:
            cur = conn.execute("SELECT status FROM gpu_jobs WHERE job_id = ?;", (job_id,))
            row = cur.fetchone()
            if not row:
                return False

            status = row["status"]
            if status == JobStatus.QUEUED.value:
                conn.execute(
                    """
                    UPDATE gpu_jobs
                    SET status = ?, finished_at = ?, progress_message = 'Cancelled before execution'
                    WHERE job_id = ?;
                    """,
                    (JobStatus.CANCELLED.value, now, job_id),
                )
                return True
            elif status == JobStatus.RUNNING.value:
                conn.execute(
                    "UPDATE gpu_jobs SET cancel_requested = 1 WHERE job_id = ?;",
                    (job_id,),
                )
                return True
            return False

    def mark_succeeded(
        self,
        job_id: str,
        result: dict[str, Any],
        progress_message: str = "Completed successfully",
    ) -> None:
        """Marks a job as SUCCEEDED with its result payload."""
        now = time.time()
        result_json = json.dumps(result, default=str)
        with self.get_connection() as conn:
            conn.execute(
                """
                UPDATE gpu_jobs
                SET status = ?, result = ?, finished_at = ?, progress_pct = 100.0,
                    progress_message = ?
                WHERE job_id = ?;
                """,
                (JobStatus.SUCCEEDED.value, result_json, now, progress_message, job_id),
            )

    def mark_failed(self, job_id: str, error_message: str) -> None:
        """Marks a job as FAILED with error traceback or reason."""
        now = time.time()
        with self.get_connection() as conn:
            conn.execute(
                """
                UPDATE gpu_jobs
                SET status = ?, error_message = ?, finished_at = ?, progress_message = 'Failed'
                WHERE job_id = ?;
                """,
                (JobStatus.FAILED.value, error_message, now, job_id),
            )

    def mark_cancelled(self, job_id: str, message: str = "Cancelled by user") -> None:
        """Marks a job as CANCELLED."""
        now = time.time()
        with self.get_connection() as conn:
            conn.execute(
                """
                UPDATE gpu_jobs
                SET status = ?, finished_at = ?, progress_message = ?
                WHERE job_id = ?;
                """,
                (JobStatus.CANCELLED.value, now, message, job_id),
            )

    def mark_interrupted(self, job_id: str, message: str = "Process interrupted") -> None:
        """Marks a job as INTERRUPTED (e.g. abrupt crash, hung worker kill, application restart)."""
        now = time.time()
        with self.get_connection() as conn:
            conn.execute(
                """
                UPDATE gpu_jobs
                SET status = ?, error_message = ?, finished_at = ?, progress_message = 'Interrupted'
                WHERE job_id = ?;
                """,
                (JobStatus.INTERRUPTED.value, message, now, job_id),
            )

    def cancel_pending_auto_jobs(self, timeframe: str | None = None) -> int:
        """Cancels all pending QUEUED auto jobs (e.g. when Stop is clicked)."""
        now = time.time()
        query = ["UPDATE gpu_jobs SET status = ?, finished_at = ?, progress_message = 'Auto job cancelled on stop' WHERE status = ? AND priority = ?"]
        params: list[Any] = [JobStatus.CANCELLED.value, now, JobStatus.QUEUED.value, JobPriority.AUTO.value]

        if timeframe is not None:
            query.append("AND timeframe = ?")
            params.append(timeframe)

        with self.get_connection() as conn:
            cur = conn.execute(" ".join(query), params)
            return cur.rowcount

    def get_state(self, key: str, default: str | None = None) -> str | None:
        """Gets a string value from scheduler_state."""
        with self.get_connection() as conn:
            cur = conn.execute("SELECT value FROM scheduler_state WHERE key = ?;", (key,))
            row = cur.fetchone()
            return row["value"] if row else default

    def set_state(self, key: str, value: str) -> None:
        """Sets a string value in scheduler_state."""
        now = time.time()
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO scheduler_state (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at;
                """,
                (key, value, now),
            )

    def get_auto_run_state(self, timeframe: str) -> AutoRunState:
        """Gets current AutoRunState for timeframe ('1h' or '4h')."""
        key = f"auto_run_state_{timeframe}"
        val = self.get_state(key, AutoRunState.STOPPED.value)
        return AutoRunState(val)

    def set_auto_run_state(self, timeframe: str, state: AutoRunState) -> None:
        """Sets current AutoRunState for timeframe."""
        key = f"auto_run_state_{timeframe}"
        self.set_state(key, state.value)

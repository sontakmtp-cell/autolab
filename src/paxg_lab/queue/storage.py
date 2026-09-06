"""SQLite persistent storage and transaction manager for GPU Queue in PAXG Forecast Lab."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sqlite3
import time
from typing import Any
import uuid

from .types import AutoRunState, JobPriority, JobSpec, JobStatus

logger = logging.getLogger(__name__)

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

                -- Partial unique index guaranteeing at most one active job per idempotency_key
                CREATE UNIQUE INDEX IF NOT EXISTS idx_active_idempotency
                ON gpu_jobs (idempotency_key)
                WHERE status IN ('QUEUED', 'RUNNING');

                CREATE TABLE IF NOT EXISTS scheduler_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS locked_verification_ledger (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timeframe TEXT NOT NULL,
                    test_start_idx INTEGER NOT NULL,
                    test_end_idx INTEGER NOT NULL,
                    test_start_time_ms INTEGER NOT NULL DEFAULT 0,
                    test_end_time_ms INTEGER NOT NULL DEFAULT 0,
                    snapshot_hash TEXT NOT NULL,
                    candidate_id TEXT NOT NULL,
                    consumed_at REAL NOT NULL,
                    verdict TEXT NOT NULL,
                    details TEXT NOT NULL DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS auto_tune_runs (
                    timeframe TEXT PRIMARY KEY,
                    snapshot_path TEXT NOT NULL,
                    snapshot_hash TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    current_trial INTEGER NOT NULL DEFAULT 0,
                    max_trials INTEGER NOT NULL DEFAULT 30,
                    best_trial_num INTEGER,
                    best_score REAL,
                    best_spec_json TEXT,
                    best_epoch INTEGER,
                    multi_seed_results_json TEXT,
                    final_candidate_id TEXT,
                    final_candidate_path TEXT,
                    last_consumed_candles INTEGER,
                    last_run_completed_at REAL,
                    current_fold INTEGER NOT NULL DEFAULT 1,
                    intermediate_fold_results_json TEXT,
                    top_specs_json TEXT,
                    multi_seed_config_idx INTEGER NOT NULL DEFAULT 0,
                    multi_seed_seed_idx INTEGER NOT NULL DEFAULT 0,
                    multi_seed_fold_idx INTEGER NOT NULL DEFAULT 1,
                    multi_seed_evaluations_json TEXT,
                    selected_test_range_json TEXT,
                    updated_at REAL NOT NULL
                );
            """)

            # Safe migration for existing locked_verification_ledger tables
            try:
                cur = conn.execute("PRAGMA table_info(locked_verification_ledger);")
                cols = [r["name"] for r in cur.fetchall()]
                if "test_start_time_ms" not in cols:
                    conn.execute("ALTER TABLE locked_verification_ledger ADD COLUMN test_start_time_ms INTEGER NOT NULL DEFAULT 0;")
                if "test_end_time_ms" not in cols:
                    conn.execute("ALTER TABLE locked_verification_ledger ADD COLUMN test_end_time_ms INTEGER NOT NULL DEFAULT 0;")
                if "details" not in cols:
                    conn.execute("ALTER TABLE locked_verification_ledger ADD COLUMN details TEXT NOT NULL DEFAULT '';")
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_locked_ledger_tf_time ON locked_verification_ledger (timeframe, test_start_time_ms, test_end_time_ms);"
                )

                # Safe migration for auto_tune_runs sub-phase columns
                cur = conn.execute("PRAGMA table_info(auto_tune_runs);")
                run_cols = [r["name"] for r in cur.fetchall()]
                for col_name, col_type in [
                    ("current_fold", "INTEGER NOT NULL DEFAULT 1"),
                    ("intermediate_fold_results_json", "TEXT"),
                    ("top_specs_json", "TEXT"),
                    ("multi_seed_config_idx", "INTEGER NOT NULL DEFAULT 0"),
                    ("multi_seed_seed_idx", "INTEGER NOT NULL DEFAULT 0"),
                    ("multi_seed_fold_idx", "INTEGER NOT NULL DEFAULT 1"),
                    ("multi_seed_evaluations_json", "TEXT"),
                    ("selected_test_range_json", "TEXT"),
                ]:
                    if col_name not in run_cols:
                        conn.execute(f"ALTER TABLE auto_tune_runs ADD COLUMN {col_name} {col_type};")

                # Backfill legacy rows in locked_verification_ledger with test_start_time_ms == 0
                cur = conn.execute("SELECT rowid, timeframe, test_start_idx, test_end_idx, snapshot_hash FROM locked_verification_ledger WHERE test_start_time_ms = 0;")
                zero_rows = cur.fetchall()
                if zero_rows:
                    snap_dir = Path("var/paxg_lab/snapshots")
                    for zrow in zero_rows:
                        shash = zrow["snapshot_hash"]
                        s_idx = zrow["test_start_idx"]
                        e_idx = zrow["test_end_idx"]
                        tf = zrow["timeframe"]
                        rid = zrow["rowid"]
                        if snap_dir.exists() and shash:
                            for spath in snap_dir.glob(f"*{tf}*"):
                                mpath = spath / "metadata.json"
                                tpath = spath / "timestamps.npy"
                                if mpath.exists() and tpath.exists():
                                    try:
                                        with open(mpath, "r", encoding="utf-8") as mf:
                                            mdata = json.load(mf)
                                        if mdata.get("sha256") == shash:
                                            import numpy as np
                                            ts = np.load(tpath)
                                            if s_idx < len(ts) and e_idx <= len(ts):
                                                t_start = int(ts[s_idx])
                                                step_ms = 3600 * 1000 if tf == "1h" else 4 * 3600 * 1000
                                                t_end = int(ts[e_idx - 1]) + step_ms
                                                conn.execute(
                                                    "UPDATE locked_verification_ledger SET test_start_time_ms = ?, test_end_time_ms = ? WHERE rowid = ?;",
                                                    (t_start, t_end, rid),
                                                )
                                                break
                                    except Exception:
                                        pass
            except Exception:
                pass

    def submit_job(self, job_spec: JobSpec, reject_if_stopped: bool = False) -> str:
        """Submits a job to the queue, enforcing atomic idempotency deduplication on active jobs."""
        if reject_if_stopped and job_spec.priority == JobPriority.AUTO.value:
            tf = job_spec.timeframe or "1h"
            if self.get_auto_run_state(tf) == AutoRunState.STOPPED:
                raise ValueError(
                    f"Cannot submit AUTO job for timeframe '{tf}': auto-run is currently STOPPED."
                )

        now = time.time()
        payload_json = json.dumps(job_spec.payload, default=str)
        result_json = json.dumps(job_spec.result, default=str) if job_spec.result is not None else None

        with self.get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE;")

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
                    conn.commit()
                    return str(row["job_id"])

            try:
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
                conn.commit()
                return job_spec.job_id
            except sqlite3.IntegrityError:
                # Concurrent insertion raced with the same active idempotency_key
                conn.rollback()
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
                raise

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

    def list_recent_jobs(
        self,
        status: str | None = None,
        job_type: str | None = None,
        timeframe: str | None = None,
        limit: int = 50,
    ) -> list[JobSpec]:
        """Convenience alias for list_jobs."""
        return self.list_jobs(status=status, job_type=job_type, timeframe=timeframe, limit=limit)

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
                cur_1h = conn.execute("SELECT value FROM scheduler_state WHERE key = 'auto_run_state_1h';").fetchone()
                cur_4h = conn.execute("SELECT value FROM scheduler_state WHERE key = 'auto_run_state_4h';").fetchone()
                state_1h = cur_1h["value"] if cur_1h else AutoRunState.SEARCHING.value
                state_4h = cur_4h["value"] if cur_4h else AutoRunState.SEARCHING.value

                ineligible_states = (AutoRunState.STOPPED.value, AutoRunState.PAUSED_ERROR.value)
                eligible_1h = (state_1h not in ineligible_states)
                eligible_4h = (state_4h not in ineligible_states)

                eligible_tfs: list[str] = []
                if eligible_1h:
                    eligible_tfs.append("1h")
                if eligible_4h:
                    eligible_tfs.append("4h")

                if eligible_tfs:
                    if last_auto_timeframe == "1h" and eligible_4h:
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
                    elif last_auto_timeframe == "4h" and eligible_1h:
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

                    # Fallback to any remaining auto job within eligible timeframes
                    if not candidate:
                        placeholders = ",".join("?" for _ in eligible_tfs)
                        cur = conn.execute(
                            f"""
                            SELECT * FROM gpu_jobs
                            WHERE status = ? AND priority = ?
                            AND (timeframe IN ({placeholders}) OR timeframe IS NULL)
                            ORDER BY created_at ASC LIMIT 1;
                            """,
                            [JobStatus.QUEUED.value, JobPriority.AUTO.value, *eligible_tfs],
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

    def get_next_queued_job(self, last_auto_timeframe: str | None = None) -> JobSpec | None:
        """Acquires the next eligible job from the queue."""
        return self.acquire_next_job(last_auto_timeframe=last_auto_timeframe)

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

    def mark_finished(
        self,
        job_id: str,
        result: dict[str, Any] | None = None,
        progress_message: str = "Completed successfully",
    ) -> None:
        """Alias for mark_succeeded."""
        self.mark_succeeded(job_id=job_id, result=result or {}, progress_message=progress_message)

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

    def mark_cancelled(
        self,
        job_id: str,
        message: str = "Cancelled by user",
        result: dict[str, Any] | None = None,
    ) -> None:
        """Marks a job as CANCELLED with optional result metadata."""
        now = time.time()
        result_json = json.dumps(result, default=str) if result is not None else None
        with self.get_connection() as conn:
            conn.execute(
                """
                UPDATE gpu_jobs
                SET status = ?, result = ?, finished_at = ?, progress_message = ?
                WHERE job_id = ?;
                """,
                (JobStatus.CANCELLED.value, result_json, now, message, job_id),
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
        """Gets current AutoRunState for timeframe ('1h' or '4h'). Default is SEARCHING."""
        key = f"auto_run_state_{timeframe}"
        val = self.get_state(key, AutoRunState.SEARCHING.value)
        return AutoRunState(val)

    def set_auto_run_state(self, timeframe: str, state: AutoRunState, allow_unstop: bool = True) -> None:
        """Sets current AutoRunState for timeframe, protecting STOPPED as a sticky user-owned state."""
        key = f"auto_run_state_{timeframe}"
        with self.get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            cur = conn.execute("SELECT value FROM scheduler_state WHERE key = ?;", (key,))
            row = cur.fetchone()
            curr_val = row["value"] if row else AutoRunState.SEARCHING.value
            if curr_val == AutoRunState.STOPPED.value and state != AutoRunState.STOPPED and not allow_unstop:
                logger.info(
                    "Ignoring state change to %s for %s: auto-run is currently STOPPED by user.",
                    state.value, timeframe,
                )
                conn.commit()
                return

            now = time.time()
            conn.execute(
                """
                INSERT INTO scheduler_state (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at;
                """,
                (key, state.value, now),
            )
            conn.commit()

    def try_acquire_coordinator_lease(
        self,
        my_pid: int,
        my_create_time: float,
        lease_timeout: float = 30.0,
        owner_token: str | None = None,
    ) -> bool:
        """Atomically attempts to claim coordinator lease in SQLite with race prevention."""
        from .process_guard import is_process_alive
        now = time.time()
        token = owner_token or f"{my_pid}_{my_create_time}_{uuid.uuid4().hex[:8]}"

        with self.get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            cur = conn.execute("SELECT value FROM scheduler_state WHERE key = 'coordinator_lease';")
            row = cur.fetchone()
            if row:
                try:
                    lease = json.loads(row["value"])
                    l_pid = int(lease.get("pid", -1))
                    l_ctime = float(lease.get("create_time", 0.0))
                    l_hb = float(lease.get("heartbeat", 0.0))

                    if l_pid != my_pid and is_process_alive(l_pid, l_ctime):
                        if (now - l_hb) < lease_timeout:
                            conn.commit()
                            return False
                except Exception:
                    pass

            new_lease = json.dumps({
                "pid": my_pid,
                "create_time": my_create_time,
                "owner_token": token,
                "heartbeat": now,
            })
            conn.execute(
                """
                INSERT INTO scheduler_state (key, value, updated_at) VALUES ('coordinator_lease', ?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at;
                """,
                (new_lease, now),
            )
            conn.commit()
            return True

    def renew_coordinator_lease(
        self,
        my_pid: int,
        my_create_time: float,
        owner_token: str | None = None,
    ) -> bool:
        """Renews coordinator lease heartbeat using compare-and-swap ownership verification.
        
        Returns:
            True if lease was renewed successfully.
            False if lease was taken over by another coordinator (split-brain prevention).
        """
        now = time.time()
        with self.get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            cur = conn.execute("SELECT value FROM scheduler_state WHERE key = 'coordinator_lease';")
            row = cur.fetchone()
            if not row:
                conn.commit()
                return False

            try:
                lease = json.loads(row["value"])
                l_pid = int(lease.get("pid", -1))
                l_ctime = float(lease.get("create_time", 0.0))
                l_token = lease.get("owner_token")

                # Ownership verification CAS
                if owner_token is not None:
                    if l_token != owner_token:
                        conn.commit()
                        return False
                else:
                    if l_pid != my_pid or abs(l_ctime - my_create_time) > 0.1:
                        conn.commit()
                        return False

                # Ownership confirmed - renew heartbeat
                token_to_keep = l_token or owner_token or f"{my_pid}_{my_create_time}"
                updated_lease = json.dumps({
                    "pid": my_pid,
                    "create_time": my_create_time,
                    "owner_token": token_to_keep,
                    "heartbeat": now,
                })
                conn.execute(
                    "UPDATE scheduler_state SET value = ?, updated_at = ? WHERE key = 'coordinator_lease';",
                    (updated_lease, now),
                )
                conn.commit()
                return True
            except Exception:
                conn.commit()
                return False

    def release_coordinator_lease(self, my_pid: int, owner_token: str | None = None) -> bool:
        """Releases coordinator lease if owned by my_pid and owner_token."""
        with self.get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            cur = conn.execute("SELECT value FROM scheduler_state WHERE key = 'coordinator_lease';")
            row = cur.fetchone()
            if row:
                try:
                    lease = json.loads(row["value"])
                    l_pid = int(lease.get("pid", -1))
                    l_token = lease.get("owner_token")
                    if owner_token is not None and l_token != owner_token:
                        conn.commit()
                        return False
                    if l_pid == my_pid:
                        conn.execute("DELETE FROM scheduler_state WHERE key = 'coordinator_lease';")
                        conn.commit()
                        return True
                except Exception:
                    pass
            conn.commit()
            return False

    def get_coordinator_lease(self) -> dict[str, Any] | None:
        """Returns the current coordinator lease data, if present."""
        with self.get_connection() as conn:
            cur = conn.execute("SELECT value FROM scheduler_state WHERE key = 'coordinator_lease';")
            row = cur.fetchone()
            if row:
                try:
                    return json.loads(row["value"])
                except Exception:
                    return None
            return None

    def get_latest_consumed_locked_range(self, timeframe: str) -> tuple[int, int] | None:
        """Returns (test_start_time_ms, test_end_time_ms) of the latest consumed locked verification range."""
        with self.get_connection() as conn:
            cur = conn.execute(
                """
                SELECT test_start_time_ms, test_end_time_ms
                FROM locked_verification_ledger
                WHERE timeframe = ? AND test_end_time_ms > 0
                ORDER BY test_end_time_ms DESC
                LIMIT 1;
                """,
                (timeframe,),
            )
            row = cur.fetchone()
            if row:
                return (int(row["test_start_time_ms"]), int(row["test_end_time_ms"]))
            return None

    def is_locked_range_consumed(
        self,
        timeframe: str,
        test_start_time_ms: int = 0,
        test_end_time_ms: int = 0,
        snapshot_hash: str | None = None,
        test_start_idx: int | None = None,
        test_end_idx: int | None = None,
    ) -> bool:
        """Checks if a candidate's locked verification interval [test_start_time_ms, test_end_time_ms)
        overlaps with ANY previously consumed interval for the given timeframe, or matches legacy unmigrated rows.
        """
        with self.get_connection() as conn:
            if test_start_time_ms > 0 and test_end_time_ms > 0:
                cur = conn.execute(
                    """
                    SELECT 1 FROM locked_verification_ledger
                    WHERE timeframe = ?
                      AND test_start_time_ms > 0
                      AND test_start_time_ms < ?
                      AND test_end_time_ms > ?;
                    """,
                    (timeframe, test_end_time_ms, test_start_time_ms),
                )
                if cur.fetchone() is not None:
                    return True

            # Fallback check on snapshot_hash + index interval overlap if timestamps were 0 or not provided
            if snapshot_hash and test_start_idx is not None and test_end_idx is not None:
                cur = conn.execute(
                    """
                    SELECT 1 FROM locked_verification_ledger
                    WHERE timeframe = ?
                      AND snapshot_hash = ?
                      AND test_start_idx < ?
                      AND test_end_idx > ?;
                    """,
                    (timeframe, snapshot_hash, test_end_idx, test_start_idx),
                )
                if cur.fetchone() is not None:
                    return True

            return False

    def record_locked_consumption(
        self,
        timeframe: str,
        test_start_idx: int,
        test_end_idx: int,
        snapshot_hash: str,
        candidate_id: str,
        test_start_time_ms: int = 0,
        test_end_time_ms: int = 0,
        verdict: str = "IN_PROGRESS",
        details: str = "",
    ) -> None:
        """Atomically records consumption of a locked verification test range into the audit ledger."""
        now = time.time()
        with self.get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            conn.execute(
                """
                INSERT INTO locked_verification_ledger (
                    timeframe, test_start_idx, test_end_idx, test_start_time_ms, test_end_time_ms,
                    snapshot_hash, candidate_id, consumed_at, verdict, details
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    timeframe,
                    test_start_idx,
                    test_end_idx,
                    test_start_time_ms,
                    test_end_time_ms,
                    snapshot_hash,
                    candidate_id,
                    now,
                    verdict,
                    details,
                ),
            )
            conn.commit()

    def update_locked_consumption_verdict(
        self,
        timeframe: str,
        candidate_id: str,
        verdict: str,
        details: str = "",
    ) -> None:
        """Updates the audit verdict and details for an evaluated candidate in locked_verification_ledger."""
        with self.get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            conn.execute(
                """
                UPDATE locked_verification_ledger
                SET verdict = ?, details = ?
                WHERE timeframe = ? AND candidate_id = ?;
                """,
                (verdict, details, timeframe, candidate_id),
            )
            conn.commit()

    def get_auto_tune_run(self, timeframe: str) -> dict[str, Any] | None:
        """Retrieves persistent state of an ongoing autonomous tuning run for timeframe."""
        with self.get_connection() as conn:
            cur = conn.execute(
                """
                SELECT * FROM auto_tune_runs WHERE timeframe = ?;
                """,
                (timeframe,),
            )
            row = cur.fetchone()
            if row:
                return dict(row)
            return None

    def save_auto_tune_run(
        self,
        timeframe: str,
        snapshot_path: str,
        snapshot_hash: str,
        phase: str,
        current_trial: int = 0,
        max_trials: int = 30,
        best_trial_num: int | None = None,
        best_score: float | None = None,
        best_spec_json: str | None = None,
        best_epoch: int | None = None,
        multi_seed_results_json: str | None = None,
        final_candidate_id: str | None = None,
        final_candidate_path: str | None = None,
        last_consumed_candles: int | None = None,
        last_run_completed_at: float | None = None,
        current_fold: int = 1,
        intermediate_fold_results_json: str | None = None,
        top_specs_json: str | None = None,
        multi_seed_config_idx: int = 0,
        multi_seed_seed_idx: int = 0,
        multi_seed_fold_idx: int = 1,
        multi_seed_evaluations_json: str | None = None,
        selected_test_range_json: str | None = None,
    ) -> None:
        """Upserts persistent state of an autonomous tuning run."""
        now = time.time()
        with self.get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            conn.execute(
                """
                INSERT INTO auto_tune_runs (
                    timeframe, snapshot_path, snapshot_hash, phase, current_trial, max_trials,
                    best_trial_num, best_score, best_spec_json, best_epoch, multi_seed_results_json,
                    final_candidate_id, final_candidate_path, last_consumed_candles, last_run_completed_at,
                    current_fold, intermediate_fold_results_json, top_specs_json,
                    multi_seed_config_idx, multi_seed_seed_idx, multi_seed_fold_idx,
                    multi_seed_evaluations_json, selected_test_range_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(timeframe) DO UPDATE SET
                    snapshot_path = excluded.snapshot_path,
                    snapshot_hash = excluded.snapshot_hash,
                    phase = excluded.phase,
                    current_trial = excluded.current_trial,
                    max_trials = excluded.max_trials,
                    best_trial_num = COALESCE(excluded.best_trial_num, auto_tune_runs.best_trial_num),
                    best_score = COALESCE(excluded.best_score, auto_tune_runs.best_score),
                    best_spec_json = COALESCE(excluded.best_spec_json, auto_tune_runs.best_spec_json),
                    best_epoch = COALESCE(excluded.best_epoch, auto_tune_runs.best_epoch),
                    multi_seed_results_json = COALESCE(excluded.multi_seed_results_json, auto_tune_runs.multi_seed_results_json),
                    final_candidate_id = COALESCE(excluded.final_candidate_id, auto_tune_runs.final_candidate_id),
                    final_candidate_path = COALESCE(excluded.final_candidate_path, auto_tune_runs.final_candidate_path),
                    last_consumed_candles = COALESCE(excluded.last_consumed_candles, auto_tune_runs.last_consumed_candles),
                    last_run_completed_at = COALESCE(excluded.last_run_completed_at, auto_tune_runs.last_run_completed_at),
                    current_fold = excluded.current_fold,
                    intermediate_fold_results_json = excluded.intermediate_fold_results_json,
                    top_specs_json = COALESCE(excluded.top_specs_json, auto_tune_runs.top_specs_json),
                    multi_seed_config_idx = excluded.multi_seed_config_idx,
                    multi_seed_seed_idx = excluded.multi_seed_seed_idx,
                    multi_seed_fold_idx = excluded.multi_seed_fold_idx,
                    multi_seed_evaluations_json = COALESCE(excluded.multi_seed_evaluations_json, auto_tune_runs.multi_seed_evaluations_json),
                    selected_test_range_json = COALESCE(excluded.selected_test_range_json, auto_tune_runs.selected_test_range_json),
                    updated_at = excluded.updated_at;
                """,
                (
                    timeframe, snapshot_path, snapshot_hash, phase, current_trial, max_trials,
                    best_trial_num, best_score, best_spec_json, best_epoch, multi_seed_results_json,
                    final_candidate_id, final_candidate_path, last_consumed_candles, last_run_completed_at,
                    current_fold, intermediate_fold_results_json, top_specs_json,
                    multi_seed_config_idx, multi_seed_seed_idx, multi_seed_fold_idx,
                    multi_seed_evaluations_json, selected_test_range_json, now,
                ),
            )
            conn.commit()

    def reset_auto_tune_run(self, timeframe: str) -> None:
        """Resets the auto tune run state for a fresh tuning cycle."""
        with self.get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            conn.execute("DELETE FROM auto_tune_runs WHERE timeframe = ?;", (timeframe,))
            conn.commit()



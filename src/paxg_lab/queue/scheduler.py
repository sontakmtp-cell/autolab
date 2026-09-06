"""Single GPU-process scheduler and job supervisor for PAXG Forecast Lab.

Enforces:
1. Strict Single GPU Worker Process invariant (never 2 GPU jobs concurrent).
2. Priority Hierarchy: Forecast > Manual > Auto Tuning.
3. 1h / 4h Alternating Round-Robin for Auto jobs.
4. Heartbeat-based hang detection and timeout enforcement.
5. Graceful stop and startup crash recovery without duplicate executions.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any

import psutil

from .process_guard import is_process_alive, safe_terminate_process
from .storage import GPUJobStorage
from .types import AutoRunState, JobPriority, JobSpec, JobStatus, JobType

logger = logging.getLogger(__name__)


class GPUScheduler:
    """Orchestrates single-process GPU job execution, heartbeats, and recovery."""

    def __init__(
        self,
        db_path: str | Path = "var/paxg_lab/paxg_lab.db",
        heartbeat_timeout: float = 30.0,
        poll_interval: float = 0.5,
        acquire_coordinator_lock: bool = True,
    ):
        self.db_path = Path(db_path)
        self.storage = GPUJobStorage(self.db_path)
        self.heartbeat_timeout = heartbeat_timeout
        self.poll_interval = poll_interval

        self.active_worker: subprocess.Popen | None = None
        self.active_job_id: str | None = None
        self.last_auto_timeframe: str | None = None

        self._stop_event = threading.Event()
        self._scheduler_thread: threading.Thread | None = None
        self._is_running = False

        if acquire_coordinator_lock:
            self._acquire_coordinator_lock()

        # Run crash recovery on initialization
        self.recovered_on_startup = self.recover_on_startup()

    def _acquire_coordinator_lock(self) -> None:
        """Acquires the scheduler coordinator singleton lock in SQLite."""
        my_pid = os.getpid()
        my_create_time = psutil.Process(my_pid).create_time()

        coord_pid_str = self.storage.get_state("coordinator_pid")
        coord_time_str = self.storage.get_state("coordinator_create_time")
        coord_hb_str = self.storage.get_state("coordinator_heartbeat")

        now = time.time()
        if coord_pid_str and coord_time_str and coord_hb_str:
            coord_pid = int(coord_pid_str)
            coord_time = float(coord_time_str)
            coord_hb = float(coord_hb_str)

            # If coordinator PID is different, alive, and beating within 30s:
            if coord_pid != my_pid and is_process_alive(coord_pid, coord_time):
                if (now - coord_hb) < 30.0:
                    raise RuntimeError(
                        f"Another GPUScheduler coordinator (PID {coord_pid}) is currently active! "
                        "Only one scheduler instance can run at a time."
                    )

        # Claim coordinator lock
        self.storage.set_state("coordinator_pid", str(my_pid))
        self.storage.set_state("coordinator_create_time", str(my_create_time))
        self.storage.set_state("coordinator_heartbeat", str(now))

    def _update_coordinator_heartbeat(self) -> None:
        """Updates scheduler coordinator heartbeat timestamp."""
        self.storage.set_state("coordinator_heartbeat", str(time.time()))

    def recover_on_startup(self) -> list[str]:
        """Recovers any orphan jobs left in RUNNING status from a previous crash or reboot."""
        recovered_ids: list[str] = []
        running_job = self.storage.get_running_job()

        if running_job:
            pid = running_job.worker_pid
            ctime = running_job.worker_create_time

            # If worker is not alive (or belongs to recycled process)
            if not is_process_alive(pid, ctime):
                logger.warning(
                    "Crash Recovery: Job '%s' was RUNNING but worker PID %s is dead. Marking INTERRUPTED.",
                    running_job.job_id,
                    pid,
                )
                self.storage.mark_interrupted(
                    running_job.job_id,
                    "Recovered on application restart: worker process died during previous session.",
                )
                recovered_ids.append(running_job.job_id)
            else:
                logger.info(
                    "Startup: Job '%s' is actively running under alive worker PID %d. Adopting job.",
                    running_job.job_id,
                    pid,
                )
                self.active_job_id = running_job.job_id

        return recovered_ids

    def tick(self) -> None:
        """Executes a single monitoring and scheduling cycle."""
        self._update_coordinator_heartbeat()

        # 1. Monitor active worker process
        if self.active_worker is not None and self.active_job_id is not None:
            ret_code = self.active_worker.poll()

            if ret_code is not None:
                # Process exited
                job = self.storage.get_job(self.active_job_id)
                if job and job.status == JobStatus.RUNNING.value:
                    # Worker process died abruptly without setting status (crash / killed)
                    logger.error(
                        "Worker process for job '%s' died abruptly with return code %d.",
                        self.active_job_id,
                        ret_code,
                    )
                    self.storage.mark_interrupted(
                        self.active_job_id,
                        f"Worker process died abruptly with return code {ret_code}.",
                    )

                self.active_worker = None
                self.active_job_id = None
            else:
                # Process is still running: inspect heartbeat and timeouts
                job = self.storage.get_job(self.active_job_id)
                if job:
                    now = time.time()

                    # Check heartbeat hang
                    if job.heartbeat_at is not None and (now - job.heartbeat_at) > self.heartbeat_timeout:
                        logger.error(
                            "Job '%s' heartbeat expired (%.1fs > %.1fs). Worker process hung! Terminating...",
                            self.active_job_id,
                            now - job.heartbeat_at,
                            self.heartbeat_timeout,
                        )
                        safe_terminate_process(job.worker_pid, job.worker_create_time, timeout=5.0)
                        self.storage.mark_failed(
                            self.active_job_id,
                            f"Heartbeat timed out after {self.heartbeat_timeout}s without response (process hung).",
                        )
                        self.active_worker = None
                        self.active_job_id = None
                        return

                    # Check job execution timeout (e.g. 20 min limit for auto jobs)
                    if job.started_at is not None and (now - job.started_at) > job.timeout_seconds:
                        logger.error(
                            "Job '%s' exceeded max timeout %.1fs. Terminating...",
                            self.active_job_id,
                            job.timeout_seconds,
                        )
                        safe_terminate_process(job.worker_pid, job.worker_create_time, timeout=5.0)
                        self.storage.mark_interrupted(
                            self.active_job_id,
                            f"Job exceeded max execution timeout of {job.timeout_seconds}s.",
                        )
                        self.active_worker = None
                        self.active_job_id = None
                        return

        # 2. Dispatch next job if no worker is active
        if self.active_worker is None:
            next_job = self.storage.acquire_next_job(last_auto_timeframe=self.last_auto_timeframe)
            if next_job:
                logger.info(
                    "Acquired next job: id='%s', type='%s', priority=%d, timeframe=%s",
                    next_job.job_id,
                    next_job.job_type,
                    next_job.priority,
                    next_job.timeframe,
                )

                if next_job.priority == JobPriority.AUTO.value and next_job.timeframe:
                    self.last_auto_timeframe = next_job.timeframe

                self._spawn_worker(next_job)

    def _spawn_worker(self, job: JobSpec) -> None:
        """Spawns worker subprocess for the acquired job."""
        cmd = [
            sys.executable,
            "-m",
            "paxg_lab.queue.worker",
            "--job-id",
            job.job_id,
            "--db-path",
            str(self.db_path),
        ]

        logger.info("Spawning worker subprocess: %s", " ".join(cmd))
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        self.active_worker = proc
        self.active_job_id = job.job_id

        # Register PID and create_time in storage
        create_time = psutil.Process(proc.pid).create_time()
        self.storage.register_worker(job.job_id, proc.pid, create_time)

    def stop_auto_run(self, timeframe: str | None = None) -> None:
        """Stops autonomous tuning: sets state to STOPPED, cancels queued auto jobs,

        and requests graceful stop on active auto job without corrupting checkpoints.
        """
        logger.info("Stopping auto run for timeframe: %s", timeframe or "all")

        # 1. Update state machine to STOPPED
        if timeframe in ("1h", "4h"):
            self.storage.set_auto_run_state(timeframe, AutoRunState.STOPPED)
        else:
            self.storage.set_auto_run_state("1h", AutoRunState.STOPPED)
            self.storage.set_auto_run_state("4h", AutoRunState.STOPPED)

        # 2. Cancel all pending queued auto jobs
        cancelled_count = self.storage.cancel_pending_auto_jobs(timeframe)
        logger.info("Cancelled %d queued auto jobs.", cancelled_count)

        # 3. If currently running job is an auto job, request graceful cancellation
        if self.active_job_id is not None:
            job = self.storage.get_job(self.active_job_id)
            if job and job.priority == JobPriority.AUTO.value:
                if timeframe is None or job.timeframe == timeframe:
                    logger.info("Requesting graceful cancellation of active auto job '%s'...", self.active_job_id)
                    self.storage.request_cancel(self.active_job_id)

    def cancel_job(self, job_id: str) -> bool:
        """Requests cancellation of a specific job."""
        return self.storage.request_cancel(job_id)

    def _loop(self) -> None:
        """Scheduler thread loop."""
        logger.info("GPUScheduler loop started.")
        while not self._stop_event.is_set():
            try:
                self.tick()
            except Exception as exc:
                logger.error("Error during scheduler tick: %s", exc)

            self._stop_event.wait(self.poll_interval)
        logger.info("GPUScheduler loop stopped.")

    def start(self) -> None:
        """Starts background scheduling loop."""
        if self._is_running:
            return
        self._is_running = True
        self._stop_event.clear()
        self._scheduler_thread = threading.Thread(
            target=self._loop,
            name="GPUScheduler-Loop",
            daemon=True,
        )
        self._scheduler_thread.start()

    def stop(self, wait_active_worker: bool = False, timeout: float = 10.0) -> None:
        """Stops scheduler thread and optionally terminates active worker."""
        self._is_running = False
        self._stop_event.set()
        if self._scheduler_thread and self._scheduler_thread.is_alive():
            self._scheduler_thread.join(timeout=3.0)

        if not wait_active_worker and self.active_worker is not None:
            if self.active_job_id:
                job = self.storage.get_job(self.active_job_id)
                if job:
                    safe_terminate_process(job.worker_pid, job.worker_create_time, timeout=timeout)
            self.active_worker = None
            self.active_job_id = None

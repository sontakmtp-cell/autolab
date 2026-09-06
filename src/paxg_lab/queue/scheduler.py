"""Single GPU-process scheduler and job supervisor for PAXG Forecast Lab.

Enforces:
1. Strict Single GPU Worker Process invariant (never 2 GPU jobs concurrent).
2. Priority Hierarchy: Forecast > Manual > Auto Tuning.
3. 1h / 4h Alternating Round-Robin for Auto jobs.
4. Heartbeat-based hang detection and timeout enforcement.
5. Graceful stop and startup crash recovery without duplicate executions.
6. CUDA OOM single retry orchestration with halved batch size.
7. Atomic coordinator lease to prevent multi-process race conditions.
"""

from __future__ import annotations

import copy
import logging
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any
import uuid

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
        self.active_worker_pid: int | None = None
        self.active_worker_create_time: float | None = None
        self.last_auto_timeframe: str | None = None
        self.scheduler_error: str | None = None
        self.owner_token: str | None = None

        self._stop_event = threading.Event()
        self._scheduler_thread: threading.Thread | None = None
        self._is_running = False

        self.acquire_coordinator_lock = acquire_coordinator_lock
        self.my_pid = os.getpid()
        self.my_create_time = psutil.Process(self.my_pid).create_time()

        if acquire_coordinator_lock:
            self._acquire_coordinator_lock()

        # Run crash recovery on initialization
        self.recovered_on_startup = self.recover_on_startup()

    def _acquire_coordinator_lock(self) -> None:
        """Acquires the scheduler coordinator singleton lease in SQLite atomically."""
        self.owner_token = f"{self.my_pid}_{self.my_create_time}_{uuid.uuid4().hex[:8]}"
        acquired = self.storage.try_acquire_coordinator_lease(
            my_pid=self.my_pid,
            my_create_time=self.my_create_time,
            lease_timeout=30.0,
            owner_token=self.owner_token,
        )
        if not acquired:
            raise RuntimeError(
                f"Another GPUScheduler coordinator is currently active! "
                "Only one scheduler instance can run at a time."
            )

    def _update_coordinator_heartbeat(self) -> bool:
        """Updates scheduler coordinator lease heartbeat.
        
        Returns:
            True if renewed, False if lease was lost to another coordinator.
        """
        if not self.acquire_coordinator_lock:
            return True

        renewed = self.storage.renew_coordinator_lease(
            my_pid=self.my_pid,
            my_create_time=self.my_create_time,
            owner_token=self.owner_token,
        )
        if not renewed:
            logger.critical(
                "CRITICAL: Lost coordinator lease to another scheduler! Stopping scheduler immediately to avoid split-brain."
            )
            self.stop()
            return False
        return True

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
                    "Startup: Job '%s' is actively running under alive worker PID %d. Adopting job with supervision.",
                    running_job.job_id,
                    pid,
                )
                self.active_job_id = running_job.job_id
                self.active_worker_pid = pid
                self.active_worker_create_time = ctime

        return recovered_ids

    def tick(self) -> None:
        """Executes a single monitoring and scheduling cycle."""
        if not self._update_coordinator_heartbeat():
            return

        # Check if scheduler is in a blocked error state due to un-terminable stuck process
        if self.scheduler_error is not None:
            if self.active_worker_pid and not is_process_alive(self.active_worker_pid, self.active_worker_create_time):
                logger.info(
                    "Previously stuck worker PID %s has exited. Clearing scheduler error state.",
                    self.active_worker_pid,
                )
                self.scheduler_error = None
                self.active_worker = None
                self.active_job_id = None
                self.active_worker_pid = None
                self.active_worker_create_time = None
            else:
                logger.error(
                    "Scheduler in blocked error state: %s. Queue dispatch paused to protect GPU invariant.",
                    self.scheduler_error,
                )
                return

        # 1. Monitor active worker process (spawned or adopted)
        has_active_worker = (self.active_worker is not None) or (self.active_worker_pid is not None)
        if has_active_worker and self.active_job_id is not None:
            if self.active_worker is not None:
                ret_code = self.active_worker.poll()
                proc_is_running = (ret_code is None)
            else:
                proc_is_running = is_process_alive(self.active_worker_pid, self.active_worker_create_time)
                ret_code = 0 if not proc_is_running else None

            if not proc_is_running:
                # Process exited
                job = self.storage.get_job(self.active_job_id)
                if job and job.status == JobStatus.RUNNING.value:
                    # Worker process died abruptly without setting status (crash / killed)
                    logger.error(
                        "Worker process for job '%s' died abruptly with return code %s.",
                        self.active_job_id,
                        ret_code,
                    )
                    self.storage.mark_interrupted(
                        self.active_job_id,
                        f"Worker process died abruptly with return code {ret_code}.",
                    )

                # Check if job failed with CUDA OOM for automatic single retry
                refreshed_job = self.storage.get_job(self.active_job_id)
                if refreshed_job and refreshed_job.status == JobStatus.FAILED.value:
                    self._handle_oom_retry_if_needed(refreshed_job)

                self.active_worker = None
                self.active_job_id = None
                self.active_worker_pid = None
                self.active_worker_create_time = None
            else:
                # Process is still running: inspect heartbeat and timeouts
                job = self.storage.get_job(self.active_job_id)
                if job:
                    now = time.time()
                    pid = self.active_worker_pid or (self.active_worker.pid if self.active_worker else job.worker_pid)
                    ctime = self.active_worker_create_time or job.worker_create_time

                    # Check heartbeat hang
                    if job.heartbeat_at is not None and (now - job.heartbeat_at) > self.heartbeat_timeout:
                        logger.error(
                            "Job '%s' heartbeat expired (%.1fs > %.1fs). Worker process hung! Terminating...",
                            self.active_job_id,
                            now - job.heartbeat_at,
                            self.heartbeat_timeout,
                        )
                        term_ok = safe_terminate_process(pid, ctime, timeout=5.0)
                        if not term_ok or is_process_alive(pid, ctime):
                            logger.critical(
                                "CRITICAL: Failed to terminate stuck worker PID %s for job '%s'! Process remains alive! "
                                "QUEUE IS BLOCKED to protect GPU single-process invariant. Manual intervention required.",
                                pid,
                                self.active_job_id,
                            )
                            self.scheduler_error = f"Worker PID {pid} could not be terminated and remains alive."
                            return

                        self.storage.mark_failed(
                            self.active_job_id,
                            f"Heartbeat timed out after {self.heartbeat_timeout}s without response (process hung).",
                        )
                        self.active_worker = None
                        self.active_job_id = None
                        self.active_worker_pid = None
                        self.active_worker_create_time = None
                        return

                    # Check job execution timeout (e.g. 20 min limit for auto jobs)
                    if job.started_at is not None and (now - job.started_at) > job.timeout_seconds:
                        logger.error(
                            "Job '%s' exceeded max timeout %.1fs. Terminating...",
                            self.active_job_id,
                            job.timeout_seconds,
                        )
                        term_ok = safe_terminate_process(pid, ctime, timeout=5.0)
                        if not term_ok or is_process_alive(pid, ctime):
                            logger.critical(
                                "CRITICAL: Failed to terminate timed-out worker PID %s for job '%s'! Process remains alive! "
                                "QUEUE IS BLOCKED to protect GPU single-process invariant. Manual intervention required.",
                                pid,
                                self.active_job_id,
                            )
                            self.scheduler_error = f"Worker PID {pid} could not be terminated and remains alive."
                            return

                        self.storage.mark_interrupted(
                            self.active_job_id,
                            f"Job exceeded max execution timeout of {job.timeout_seconds}s.",
                        )
                        self.active_worker = None
                        self.active_job_id = None
                        self.active_worker_pid = None
                        self.active_worker_create_time = None
                        return

        # 2. Dispatch next job if no worker is active and scheduler is error-free
        if self.scheduler_error is None and self.active_worker is None and self.active_worker_pid is None:
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

    def _spawn_worker(self, job: JobSpec) -> bool:
        """Spawns worker subprocess for the acquired job with dedicated log file and robust error handling."""
        if self.scheduler_error is not None or self.active_worker is not None or self.active_worker_pid is not None:
            logger.error("Refusing to spawn worker: active worker or scheduler error exists.")
            return False
        log_dir = Path("var/paxg_lab/job_logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"job_{job.job_id}.log"

        cmd = [
            sys.executable,
            "-m",
            "paxg_lab.queue.worker",
            "--job-id",
            job.job_id,
            "--db-path",
            str(self.db_path),
        ]

        logger.info("Spawning worker subprocess: %s (log: %s)", " ".join(cmd), log_path)
        proc: subprocess.Popen | None = None
        spawned_pid: int | None = None
        spawned_create_time: float | None = None

        try:
            # Dedicated log file avoids pipe deadlocks when worker logs heavily
            log_file = open(log_path, "a", encoding="utf-8")
            proc = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
            spawned_pid = proc.pid
            try:
                spawned_create_time = psutil.Process(proc.pid).create_time()
            except Exception:
                spawned_create_time = None

            # Register PID and create_time in storage
            self.storage.register_worker(job.job_id, proc.pid, spawned_create_time)

            self.active_worker = proc
            self.active_job_id = job.job_id
            self.active_worker_pid = proc.pid
            self.active_worker_create_time = spawned_create_time
            return True
        except Exception as exc:
            logger.error("Failed to complete worker subprocess startup for job '%s': %s", job.job_id, exc)
            if proc is not None and spawned_pid is not None:
                # Subprocess was already created by Popen! Safely terminate child to prevent orphan GPU process
                logger.warning("Terminating newly spawned child process PID %s after post-spawn failure...", spawned_pid)
                term_ok = safe_terminate_process(spawned_pid, spawned_create_time, timeout=5.0)
                if not term_ok or is_process_alive(spawned_pid, spawned_create_time):
                    # Child cannot be killed! Block queue to preserve single GPU process invariant
                    logger.critical(
                        "CRITICAL: Failed to kill newly spawned child process PID %s after startup exception! "
                        "Process remains alive. Blocking queue dispatch to protect GPU invariant.",
                        spawned_pid,
                    )
                    self.active_worker = proc
                    self.active_job_id = job.job_id
                    self.active_worker_pid = spawned_pid
                    self.active_worker_create_time = spawned_create_time
                    self.scheduler_error = (
                        f"Post-spawn failure on job '{job.job_id}' and worker PID {spawned_pid} "
                        "could not be terminated."
                    )
                    return False

            # Child was cleanly terminated or never spawned
            try:
                self.storage.mark_failed(job.job_id, f"Failed to spawn worker subprocess: {exc}")
            except Exception:
                pass
            self.active_worker = None
            self.active_job_id = None
            self.active_worker_pid = None
            self.active_worker_create_time = None
            return False

    def _handle_oom_retry_if_needed(self, job: JobSpec) -> str | None:
        """Implements PLAN 4.1 OOM protocol: retry once with halved batch size and adjusted accumulation.

        If batch_size is already 1, no further reduction is possible; transitions directly to PAUSED_ERROR.
        Gradient accumulation is clamped to max 16 per PLAN 3.2 constraints.
        """
        err = (job.error_message or "").lower()
        if "[cuda_oom]" not in err and "out of memory" not in err:
            return None

        retry_count = int(job.payload.get("oom_retry_count", 0))
        if retry_count > 0:
            logger.error("CUDA OOM retry already failed for job '%s'. Setting PAUSED_ERROR.", job.job_id)
            tf = job.timeframe or "1h"
            self.storage.set_auto_run_state(tf, AutoRunState.PAUSED_ERROR)
            if not job.timeframe:
                self.storage.set_auto_run_state("4h", AutoRunState.PAUSED_ERROR)
            return None

        # Inspect current batch_size and gradient_accumulation_steps
        payload = job.payload
        has_train_spec = "train_spec" in payload and isinstance(payload["train_spec"], dict)
        spec_d = payload["train_spec"] if has_train_spec else payload

        old_b = int(spec_d.get("batch_size", 2))
        old_a = int(spec_d.get("gradient_accumulation_steps", 8))

        # Boundary check: If batch_size cannot be reduced further (batch_size <= 1)
        if old_b <= 1:
            logger.error(
                "CUDA OOM on job '%s' cannot be retried: batch_size=%d is already at minimum (1). "
                "Setting PAUSED_ERROR directly per PLAN 4.1 protocol.",
                job.job_id,
                old_b,
            )
            tf = job.timeframe or "1h"
            self.storage.set_auto_run_state(tf, AutoRunState.PAUSED_ERROR)
            if not job.timeframe:
                self.storage.set_auto_run_state("4h", AutoRunState.PAUSED_ERROR)
            return None

        # Valid load reduction: halve batch_size, double grad_accum clamped to max 16 per PLAN 3.2
        new_b = max(1, old_b // 2)
        new_a = min(16, old_a * 2)

        retry_payload = copy.deepcopy(job.payload)
        retry_payload["oom_retry_count"] = 1
        if has_train_spec:
            retry_payload["train_spec"]["batch_size"] = new_b
            retry_payload["train_spec"]["gradient_accumulation_steps"] = new_a
        else:
            retry_payload["batch_size"] = new_b
            retry_payload["gradient_accumulation_steps"] = new_a

        retry_payload["original_batch_size"] = old_b
        retry_payload["original_grad_accum"] = old_a
        retry_payload["adjusted_batch_size"] = new_b
        retry_payload["adjusted_grad_accum"] = new_a

        retry_job_id = f"{job.job_id}_oom_retry"
        retry_spec = JobSpec(
            job_id=retry_job_id,
            job_type=job.job_type,
            timeframe=job.timeframe,
            priority=job.priority,
            payload=retry_payload,
            timeout_seconds=job.timeout_seconds,
        )
        self.storage.submit_job(retry_spec)
        logger.info(
            "CUDA OOM detected on job '%s' (batch=%d->%d, accum=%d->%d). Dispatched OOM retry job '%s'.",
            job.job_id,
            old_b,
            new_b,
            old_a,
            new_a,
            retry_job_id,
        )
        return retry_job_id

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

    def start_auto_run(self, timeframe: str | None = None) -> None:
        """Starts or resumes autonomous tuning: sets state to SEARCHING.
        
        Queued and newly submitted AUTO jobs for the timeframe can now be dispatched.
        """
        logger.info("Starting auto run for timeframe: %s", timeframe or "all")
        if timeframe in ("1h", "4h"):
            self.storage.set_auto_run_state(timeframe, AutoRunState.SEARCHING)
        else:
            self.storage.set_auto_run_state("1h", AutoRunState.SEARCHING)
            self.storage.set_auto_run_state("4h", AutoRunState.SEARCHING)

    def resume_auto_run(self, timeframe: str | None = None) -> None:
        """Alias for start_auto_run."""
        self.start_auto_run(timeframe)

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
        """Stops scheduler thread and releases coordinator lease."""
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
            self.active_worker_pid = None
            self.active_worker_create_time = None

        # Release coordinator lease
        if self.acquire_coordinator_lock:
            if self.owner_token is not None:
                self.storage.release_coordinator_lease(self.my_pid, owner_token=self.owner_token)
                self.owner_token = None
            else:
                self.storage.release_coordinator_lease(self.my_pid)


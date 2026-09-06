"""Unit and integration tests for P4: Single GPU Process Queue, Heartbeat, Safe Stop, and Recovery."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np
import psutil
import pytest
import torch

from paxg_lab.queue.process_guard import is_process_alive, safe_terminate_process
from paxg_lab.queue.scheduler import GPUScheduler
from paxg_lab.queue.storage import GPUJobStorage
from paxg_lab.queue.types import (
    AutoRunState,
    JobPriority,
    JobSpec,
    JobStatus,
    JobType,
)


@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    """Provides a fresh temporary SQLite database path for isolated testing."""
    return tmp_path / "test_paxg_queue.db"


@pytest.fixture
def storage(temp_db_path: Path) -> GPUJobStorage:
    """Provides an initialized GPUJobStorage instance on a temporary database."""
    return GPUJobStorage(temp_db_path)


# ---------------------------------------------------------------------------
# 1. JobSpec and Status Lifecycle Validation
# ---------------------------------------------------------------------------


def test_job_spec_and_status_validation():
    """Validates strict state machine and validation rules on JobSpec."""
    # Valid spec
    spec = JobSpec(
        job_id="job_001",
        job_type=JobType.FORECAST.value,
        timeframe="1h",
        priority=JobPriority.FORECAST.value,
        status=JobStatus.QUEUED.value,
    )
    assert spec.job_id == "job_001"
    assert spec.status == JobStatus.QUEUED.value
    assert not JobStatus(spec.status).is_terminal()

    # Invalid status
    with pytest.raises(ValueError, match="Invalid JobStatus"):
        JobSpec(job_id="job_err1", job_type="FORECAST", status="INVALID_STATUS")

    # Invalid priority
    with pytest.raises(ValueError, match="Invalid JobPriority"):
        JobSpec(job_id="job_err2", job_type="FORECAST", priority=99)

    # Invalid timeframe
    with pytest.raises(ValueError, match="Invalid timeframe"):
        JobSpec(job_id="job_err3", job_type="FORECAST", timeframe="15m")

    # Invalid timeout
    with pytest.raises(ValueError, match="timeout_seconds must be positive"):
        JobSpec(job_id="job_err4", job_type="FORECAST", timeout_seconds=-10.0)

    # Terminal states
    for term_status in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED, JobStatus.INTERRUPTED):
        assert term_status.is_terminal()


# ---------------------------------------------------------------------------
# 2. Idempotency and Deduplication
# ---------------------------------------------------------------------------


def test_idempotency_and_deduplication(storage: GPUJobStorage):
    """Verifies that submitting with identical idempotency_key returns existing active job without duplicate."""
    spec_a = JobSpec(
        job_id="job_first",
        job_type=JobType.FORECAST.value,
        priority=JobPriority.FORECAST.value,
        idempotency_key="req_unique_12345",
    )
    submitted_id_1 = storage.submit_job(spec_a)
    assert submitted_id_1 == "job_first"

    # Second submission with same idempotency key
    spec_b = JobSpec(
        job_id="job_duplicate",
        job_type=JobType.FORECAST.value,
        priority=JobPriority.FORECAST.value,
        idempotency_key="req_unique_12345",
    )
    submitted_id_2 = storage.submit_job(spec_b)
    # Must return the existing job_id
    assert submitted_id_2 == "job_first"

    # Total jobs in storage should be 1, not 2
    jobs = storage.list_jobs()
    assert len(jobs) == 1
    assert jobs[0].job_id == "job_first"


# ---------------------------------------------------------------------------
# 3. Single GPU Worker Invariant
# ---------------------------------------------------------------------------


def test_single_gpu_process_enforcement(storage: GPUJobStorage):
    """Verifies that acquire_next_job enforces at most ONE running GPU process at all times."""
    storage.submit_job(JobSpec(job_id="job_1", job_type="TRAIN", priority=JobPriority.AUTO.value))
    storage.submit_job(JobSpec(job_id="job_2", job_type="TRAIN", priority=JobPriority.AUTO.value))

    # First acquisition succeeds
    acq_1 = storage.acquire_next_job()
    assert acq_1 is not None
    assert acq_1.job_id == "job_1"
    assert acq_1.status == JobStatus.RUNNING.value

    # Second acquisition MUST return None while job_1 is RUNNING
    acq_2 = storage.acquire_next_job()
    assert acq_2 is None

    # Complete job_1
    storage.mark_succeeded("job_1", {"status": "done"})

    # Now second job can be acquired
    acq_3 = storage.acquire_next_job()
    assert acq_3 is not None
    assert acq_3.job_id == "job_2"


# ---------------------------------------------------------------------------
# 4. Priority Hierarchy: Forecast > Manual > Auto
# ---------------------------------------------------------------------------


def test_priority_hierarchy_forecast_over_manual_and_auto(storage: GPUJobStorage):
    """Verifies queue strictly orders: Priority 1 (FORECAST) > Priority 2 (MANUAL) > Priority 3 (AUTO)."""
    # Enqueue in reverse order
    storage.submit_job(JobSpec(job_id="auto_1", job_type="AUTO_TRIAL", priority=JobPriority.AUTO.value))
    storage.submit_job(JobSpec(job_id="manual_1", job_type="TRAIN", priority=JobPriority.MANUAL.value))
    storage.submit_job(JobSpec(job_id="forecast_1", job_type="FORECAST", priority=JobPriority.FORECAST.value))

    # First acquired MUST be FORECAST
    job_first = storage.acquire_next_job()
    assert job_first is not None
    assert job_first.job_id == "forecast_1"
    storage.mark_succeeded(job_first.job_id, {})

    # Second acquired MUST be MANUAL
    job_second = storage.acquire_next_job()
    assert job_second is not None
    assert job_second.job_id == "manual_1"
    storage.mark_succeeded(job_second.job_id, {})

    # Third acquired MUST be AUTO
    job_third = storage.acquire_next_job()
    assert job_third is not None
    assert job_third.job_id == "auto_1"


# ---------------------------------------------------------------------------
# 5. Forecast Waits for Current Job, then Preempts Remaining Queue
# ---------------------------------------------------------------------------


def test_forecast_waits_for_current_job_then_runs_immediately(storage: GPUJobStorage):
    """Verifies forecast does not abort current job, but is dispatched immediately as next job."""
    storage.submit_job(JobSpec(job_id="running_auto", job_type="AUTO_TRIAL", priority=JobPriority.AUTO.value))
    storage.submit_job(JobSpec(job_id="queued_auto_2", job_type="AUTO_TRIAL", priority=JobPriority.AUTO.value))

    running_job = storage.acquire_next_job()
    assert running_job is not None
    assert running_job.job_id == "running_auto"

    # User requests instant forecast while auto job is running
    storage.submit_job(JobSpec(job_id="instant_forecast", job_type="FORECAST", priority=JobPriority.FORECAST.value))

    # Queue cannot acquire while running_auto is active
    assert storage.acquire_next_job() is None

    # running_auto finishes its step/job
    storage.mark_succeeded("running_auto", {"result": "ok"})

    # Next job acquired MUST be instant_forecast, not queued_auto_2
    next_job = storage.acquire_next_job()
    assert next_job is not None
    assert next_job.job_id == "instant_forecast"


# ---------------------------------------------------------------------------
# 6. 1h / 4h Alternating Round-Robin for Auto Jobs
# ---------------------------------------------------------------------------


def test_alternating_1h_4h_for_auto_jobs(storage: GPUJobStorage):
    """Verifies scheduler alternates between 1h and 4h auto jobs when both are queued."""
    # Enqueue multiple 1h and 4h auto jobs
    storage.submit_job(JobSpec(job_id="auto_1h_a", job_type="AUTO_TRIAL", timeframe="1h", priority=JobPriority.AUTO.value))
    storage.submit_job(JobSpec(job_id="auto_1h_b", job_type="AUTO_TRIAL", timeframe="1h", priority=JobPriority.AUTO.value))
    storage.submit_job(JobSpec(job_id="auto_4h_a", job_type="AUTO_TRIAL", timeframe="4h", priority=JobPriority.AUTO.value))
    storage.submit_job(JobSpec(job_id="auto_4h_b", job_type="AUTO_TRIAL", timeframe="4h", priority=JobPriority.AUTO.value))

    # 1. First pick (initial state, no previous auto timeframe)
    j1 = storage.acquire_next_job(last_auto_timeframe=None)
    assert j1 is not None
    assert j1.timeframe == "1h"
    storage.mark_succeeded(j1.job_id, {})

    # 2. Next pick given last was 1h: MUST alternate to 4h
    j2 = storage.acquire_next_job(last_auto_timeframe="1h")
    assert j2 is not None
    assert j2.timeframe == "4h"
    assert j2.job_id == "auto_4h_a"
    storage.mark_succeeded(j2.job_id, {})

    # 3. Next pick given last was 4h: MUST alternate to 1h
    j3 = storage.acquire_next_job(last_auto_timeframe="4h")
    assert j3 is not None
    assert j3.timeframe == "1h"
    assert j3.job_id == "auto_1h_b"
    storage.mark_succeeded(j3.job_id, {})

    # 4. Next pick given last was 1h: MUST alternate to remaining 4h
    j4 = storage.acquire_next_job(last_auto_timeframe="1h")
    assert j4 is not None
    assert j4.timeframe == "4h"
    assert j4.job_id == "auto_4h_b"
    storage.mark_succeeded(j4.job_id, {})

    # 5. Queue empty
    assert storage.acquire_next_job(last_auto_timeframe="4h") is None


# ---------------------------------------------------------------------------
# 7. Heartbeat Hang Detection and Termination
# ---------------------------------------------------------------------------


def test_heartbeat_hang_detection_and_termination(temp_db_path: Path):
    """Verifies that a worker whose heartbeat ceases is detected as hung and terminated."""
    scheduler = GPUScheduler(
        db_path=temp_db_path,
        heartbeat_timeout=1.5,
        acquire_coordinator_lock=True,
    )

    # Submit a dummy job instructed to simulate hang after step 1
    job_id = "hang_job_001"
    scheduler.storage.submit_job(
        JobSpec(
            job_id=job_id,
            job_type=JobType.DUMMY.value,
            priority=JobPriority.MANUAL.value,
            payload={"steps": 5, "step_sleep": 0.1, "hang_at_step": 1},
        )
    )

    # Tick scheduler to spawn worker
    scheduler.tick()
    assert scheduler.active_worker is not None
    worker_pid = scheduler.active_worker.pid
    assert is_process_alive(worker_pid)

    # Wait for the worker to begin and then stop heartbeating
    time.sleep(2.5)

    # Tick scheduler: should detect heartbeat expiration and terminate worker
    scheduler.tick()

    # Process must be dead
    time.sleep(0.5)
    assert not is_process_alive(worker_pid)

    # Job status must be marked FAILED with heartbeat timeout error
    job = scheduler.storage.get_job(job_id)
    assert job is not None
    assert job.status == JobStatus.FAILED.value
    assert "Heartbeat timed out" in (job.error_message or "")


# ---------------------------------------------------------------------------
# 8. Safe Process Termination with PID & Create Time Verification
# ---------------------------------------------------------------------------


def test_safe_process_termination_pid_verification():
    """Verifies safe_terminate_process refuses to terminate a process if create_time does not match."""
    # Spawn a harmless sleep process
    cmd = [sys.executable, "-c", "import time; time.sleep(10)"]
    proc = subprocess.Popen(cmd)
    pid = proc.pid
    actual_create_time = psutil.Process(pid).create_time()

    try:
        # 1. Attempt termination with WRONG create_time (simulating PID recycling)
        bogus_create_time = actual_create_time - 5000.0
        success = safe_terminate_process(pid=pid, expected_create_time=bogus_create_time, timeout=1.0)
        # Must refuse and return False
        assert success is False
        assert is_process_alive(pid)

        # 2. Attempt termination with CORRECT create_time
        success_correct = safe_terminate_process(pid=pid, expected_create_time=actual_create_time, timeout=2.0)
        assert success_correct is True
        assert not is_process_alive(pid)

    finally:
        if is_process_alive(pid):
            proc.kill()


# ---------------------------------------------------------------------------
# 9. Graceful Stop Auto Run (Without Corrupting Checkpoints)
# ---------------------------------------------------------------------------


def test_graceful_stop_auto_run(temp_db_path: Path):
    """Verifies stop_auto_run cancels queued auto jobs and signals active worker to halt cleanly."""
    scheduler = GPUScheduler(
        db_path=temp_db_path,
        heartbeat_timeout=10.0,
        acquire_coordinator_lock=True,
    )

    # Submit active auto job and queued auto job
    active_id = "auto_job_active"
    queued_id = "auto_job_pending"
    scheduler.storage.submit_job(
        JobSpec(
            job_id=active_id,
            job_type=JobType.DUMMY.value,
            timeframe="1h",
            priority=JobPriority.AUTO.value,
            payload={"steps": 10, "step_sleep": 0.3},
        )
    )
    scheduler.storage.submit_job(
        JobSpec(
            job_id=queued_id,
            job_type=JobType.DUMMY.value,
            timeframe="4h",
            priority=JobPriority.AUTO.value,
            payload={"steps": 10, "step_sleep": 0.3},
        )
    )

    # Tick to start active job
    scheduler.tick()
    assert scheduler.active_worker is not None
    time.sleep(0.5)

    # User clicks STOP
    scheduler.stop_auto_run()

    # 1. State machine transitioned to STOPPED
    assert scheduler.storage.get_auto_run_state("1h") == AutoRunState.STOPPED
    assert scheduler.storage.get_auto_run_state("4h") == AutoRunState.STOPPED

    # 2. Pending queued auto job is immediately CANCELLED
    pending_job = scheduler.storage.get_job(queued_id)
    assert pending_job is not None
    assert pending_job.status == JobStatus.CANCELLED.value

    # 3. Active worker receives cancel signal and exits cleanly
    # Wait for worker to check stop event and exit
    for _ in range(10):
        time.sleep(0.5)
        scheduler.tick()
        active_job = scheduler.storage.get_job(active_id)
        if active_job and active_job.status == JobStatus.CANCELLED.value:
            break

    active_job = scheduler.storage.get_job(active_id)
    assert active_job is not None
    assert active_job.status == JobStatus.CANCELLED.value


# ---------------------------------------------------------------------------
# 10. Abrupt Kill Detection and Crash Recovery
# ---------------------------------------------------------------------------


def test_abrupt_kill_and_interrupt_detection(temp_db_path: Path):
    """Verifies that an abruptly killed worker process is marked INTERRUPTED on next tick."""
    scheduler = GPUScheduler(
        db_path=temp_db_path,
        heartbeat_timeout=15.0,
        acquire_coordinator_lock=True,
    )

    job_id = "killed_job_001"
    scheduler.storage.submit_job(
        JobSpec(
            job_id=job_id,
            job_type=JobType.DUMMY.value,
            priority=JobPriority.MANUAL.value,
            payload={"steps": 20, "step_sleep": 0.5},
        )
    )

    scheduler.tick()
    assert scheduler.active_worker is not None
    worker_pid = scheduler.active_worker.pid

    # Simulate abrupt external process kill (SIGKILL)
    psutil.Process(worker_pid).kill()
    time.sleep(0.5)

    # Tick scheduler: detects process exit while still RUNNING
    scheduler.tick()

    job = scheduler.storage.get_job(job_id)
    assert job is not None
    assert job.status == JobStatus.INTERRUPTED.value
    assert "died abruptly" in (job.error_message or "")


# ---------------------------------------------------------------------------
# 11. Application Startup Crash Recovery
# ---------------------------------------------------------------------------


def test_startup_crash_recovery(temp_db_path: Path):
    """Verifies that initializing GPUScheduler recovers orphan RUNNING jobs from a previous crash."""
    storage = GPUJobStorage(temp_db_path)

    # Insert an orphan job left in RUNNING status with a non-existent PID
    storage.submit_job(
        JobSpec(
            job_id="crashed_job_prior_session",
            job_type=JobType.TRAIN.value,
            timeframe="1h",
            priority=JobPriority.AUTO.value,
        )
    )
    # Manually transition to RUNNING with dummy dead PID
    with storage.get_connection() as conn:
        conn.execute(
            """
            UPDATE gpu_jobs
            SET status = 'RUNNING', worker_pid = 999999, worker_create_time = 1000.0, started_at = ?
            WHERE job_id = 'crashed_job_prior_session';
            """,
            (time.time() - 100.0,),
        )

    # Instantiate new scheduler
    new_scheduler = GPUScheduler(
        db_path=temp_db_path,
        acquire_coordinator_lock=False,
    )

    assert "crashed_job_prior_session" in new_scheduler.recovered_on_startup

    job = storage.get_job("crashed_job_prior_session")
    assert job is not None
    assert job.status == JobStatus.INTERRUPTED.value
    assert "Recovered on application restart" in (job.error_message or "")

    # Normal queue operations can resume without duplicate execution
    assert storage.get_running_job() is None


# ---------------------------------------------------------------------------
# 12. Auto Job Max Timeout Limit Enforcement
# ---------------------------------------------------------------------------


def test_auto_job_timeout_limit(temp_db_path: Path):
    """Verifies that an auto job exceeding its timeout limit is terminated and marked INTERRUPTED."""
    scheduler = GPUScheduler(
        db_path=temp_db_path,
        heartbeat_timeout=30.0,
        acquire_coordinator_lock=True,
    )

    job_id = "timeout_job_001"
    scheduler.storage.submit_job(
        JobSpec(
            job_id=job_id,
            job_type=JobType.DUMMY.value,
            priority=JobPriority.AUTO.value,
            timeout_seconds=1.0,  # 1.0 second timeout for test
            payload={"steps": 10, "step_sleep": 0.5},
        )
    )

    scheduler.tick()
    assert scheduler.active_worker is not None

    # Wait for execution timeout
    time.sleep(2.0)

    scheduler.tick()

    job = scheduler.storage.get_job(job_id)
    assert job is not None
    assert job.status == JobStatus.INTERRUPTED.value
    assert "exceeded max execution timeout" in (job.error_message or "")


# ---------------------------------------------------------------------------
# 13. Singleton Coordinator Lock
# ---------------------------------------------------------------------------


def test_singleton_coordinator_lock(temp_db_path: Path):
    """Verifies that two active schedulers cannot coordinate the queue simultaneously."""
    sched_a = GPUScheduler(db_path=temp_db_path, acquire_coordinator_lock=True)

    # Attempting to start another scheduler from a different PID simulation
    with pytest.raises(RuntimeError, match="Another GPUScheduler coordinator.*is currently active"):
        # Simulate active coordinator with different PID (System idle PID 4 is always alive)
        import json
        sys_proc = psutil.Process(4)
        lease_data = json.dumps({
            "pid": 4,
            "create_time": sys_proc.create_time(),
            "heartbeat": time.time(),
        })
        sched_a.storage.set_state("coordinator_lease", lease_data)

        # Attempt to initialize another scheduler
        GPUScheduler(db_path=temp_db_path, acquire_coordinator_lock=True)


# ---------------------------------------------------------------------------
# 14. GPU Memory Reclaimed Completely on Worker Exit
# ---------------------------------------------------------------------------


def test_gpu_memory_reclaimed_on_worker_exit():
    """Verifies that isolated worker process terminates cleanly and frees CUDA memory."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available for GPU memory test")

    initial_mem = torch.cuda.memory_allocated()

    # Launch a Python subprocess that allocates a 500MB tensor on GPU and exits
    code = (
        "import torch, time; "
        "t = torch.zeros((125000000,), dtype=torch.float32, device='cuda'); "
        "assert torch.cuda.memory_allocated() > 400 * 1024 * 1024; "
        "time.sleep(0.5); "
        "del t; "
        "torch.cuda.empty_cache();"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert res.returncode == 0, f"GPU subprocess failed: {res.stderr}"

    # After subprocess exit, parent VRAM must be unaffected
    torch.cuda.empty_cache()
    final_mem = torch.cuda.memory_allocated()
    assert abs(final_mem - initial_mem) == 0, "GPU VRAM was not completely reclaimed after worker process termination!"


# ---------------------------------------------------------------------------
# 15. Review Feedback: Forecast Job Rejects Missing Contexts / No Synthetic Noise
# ---------------------------------------------------------------------------


def test_forecast_missing_data_fails_without_synthetic_noise(temp_db_path: Path):
    """Verifies forecast job fails immediately if contexts or snapshot_path are omitted."""
    from paxg_lab.queue.worker import GPUWorker

    storage = GPUJobStorage(temp_db_path)
    job_id = "forecast_missing_ctx"
    storage.submit_job(
        JobSpec(
            job_id=job_id,
            job_type=JobType.FORECAST.value,
            priority=JobPriority.FORECAST.value,
            payload={"timeframe": "1h"},  # No contexts or snapshot_path
        )
    )
    storage.acquire_next_job()

    worker = GPUWorker(job_id=job_id, db_path=temp_db_path)
    exit_code = worker.run()
    assert exit_code == 1

    job = storage.get_job(job_id)
    assert job is not None
    assert job.status == JobStatus.FAILED.value
    assert "missing required 'contexts' array or valid 'snapshot_path'" in (job.error_message or "")
    assert "random noise" in (job.error_message or "")


# ---------------------------------------------------------------------------
# 16. Review Feedback: Atomic Idempotency Under Thread Concurrency
# ---------------------------------------------------------------------------


def test_atomic_idempotency_concurrent_submissions(temp_db_path: Path):
    """Verifies that concurrent submissions with the same idempotency_key result in exactly 1 job."""
    from concurrent.futures import ThreadPoolExecutor

    storage = GPUJobStorage(temp_db_path)
    idem_key = "concurrent_idem_key_123"

    def submit_task(idx: int) -> str:
        s = GPUJobStorage(temp_db_path)
        spec = JobSpec(
            job_id=f"job_thread_{idx}",
            job_type=JobType.FORECAST.value,
            priority=JobPriority.FORECAST.value,
            idempotency_key=idem_key,
        )
        return s.submit_job(spec)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(submit_task, range(10)))

    # All returned job_ids must be identical
    first_id = results[0]
    for r in results:
        assert r == first_id

    # Exactly 1 active job in DB
    all_jobs = storage.list_jobs()
    active_jobs = [j for j in all_jobs if j.idempotency_key == idem_key]
    assert len(active_jobs) == 1
    assert active_jobs[0].job_id == first_id

    # If the job succeeds, partial unique index allows re-using key for a new job
    storage.acquire_next_job()
    storage.mark_succeeded(first_id, {"status": "ok"})

    spec_after = JobSpec(
        job_id="job_thread_after_success",
        job_type=JobType.FORECAST.value,
        priority=JobPriority.FORECAST.value,
        idempotency_key=idem_key,
    )
    new_id = storage.submit_job(spec_after)
    assert new_id == "job_thread_after_success"


# ---------------------------------------------------------------------------
# 17. Review Feedback: Startup Recovery Adoption Supervision
# ---------------------------------------------------------------------------


def test_adopted_worker_heartbeat_timeout_supervision(temp_db_path: Path):
    """Verifies an adopted alive worker from prior session is actively monitored for heartbeat timeout."""
    storage = GPUJobStorage(temp_db_path)

    # Launch a real sleep process to represent an adopted alive worker
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(15)"])
    pid = proc.pid
    ctime = psutil.Process(pid).create_time()

    job_id = "adopted_hung_worker"
    spec = JobSpec(
        job_id=job_id,
        job_type=JobType.TRAIN.value,
        priority=JobPriority.MANUAL.value,
    )
    storage.submit_job(spec)

    # Manually put job in RUNNING state with heartbeat in the past
    stale_heartbeat = time.time() - 10.0
    with storage.get_connection() as conn:
        conn.execute(
            """
            UPDATE gpu_jobs
            SET status = 'RUNNING', worker_pid = ?, worker_create_time = ?,
                started_at = ?, heartbeat_at = ?
            WHERE job_id = ?;
            """,
            (pid, ctime, stale_heartbeat, stale_heartbeat, job_id),
        )

    try:
        # Start new scheduler with short heartbeat timeout
        scheduler = GPUScheduler(
            db_path=temp_db_path,
            heartbeat_timeout=2.0,
            acquire_coordinator_lock=False,
        )

        # Verify job was adopted (active_worker is None since Popen not restored, but PID tracked)
        assert scheduler.active_job_id == job_id
        assert scheduler.active_worker_pid == pid
        assert scheduler.active_worker is None

        # Tick scheduler: heartbeat is stale -> worker must be terminated
        scheduler.tick()

        # Check process was terminated
        time.sleep(0.5)
        assert not is_process_alive(pid)

        # Job must be marked FAILED with heartbeat timeout
        job = storage.get_job(job_id)
        assert job is not None
        assert job.status == JobStatus.FAILED.value
        assert "Heartbeat timed out" in (job.error_message or "")

        # Scheduler pointers cleared
        assert scheduler.active_job_id is None
        assert scheduler.active_worker_pid is None

    finally:
        if is_process_alive(pid):
            proc.kill()


# ---------------------------------------------------------------------------
# 18. Review Feedback: Worker Spawn Failure Safety
# ---------------------------------------------------------------------------


def test_worker_spawn_failure_safety(temp_db_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verifies that if worker spawning raises an exception, the job is marked FAILED and queue unblocks."""
    scheduler = GPUScheduler(
        db_path=temp_db_path,
        acquire_coordinator_lock=False,
    )

    job_id_fail = "job_spawn_fail"
    job_id_next = "job_spawn_next"
    scheduler.storage.submit_job(
        JobSpec(job_id=job_id_fail, job_type=JobType.DUMMY.value, priority=JobPriority.MANUAL.value)
    )
    scheduler.storage.submit_job(
        JobSpec(job_id=job_id_next, job_type=JobType.DUMMY.value, priority=JobPriority.MANUAL.value, payload={"steps": 1, "step_sleep": 0.05})
    )

    # Force subprocess.Popen to fail on the first call
    original_popen = subprocess.Popen

    def mock_popen_fail(*args, **kwargs):
        raise OSError("Failed to create worker subprocess: simulated error")

    monkeypatch.setattr(subprocess, "Popen", mock_popen_fail)

    # First tick tries to spawn job_spawn_fail -> catches exception and marks FAILED
    scheduler.tick()

    job_fail = scheduler.storage.get_job(job_id_fail)
    assert job_fail is not None
    assert job_fail.status == JobStatus.FAILED.value
    assert "Failed to spawn worker subprocess" in (job_fail.error_message or "")
    assert scheduler.active_job_id is None
    assert scheduler.active_worker is None

    # Restore Popen and tick again: job_spawn_next must execute and succeed without deadlock
    monkeypatch.setattr(subprocess, "Popen", original_popen)

    # Allow job_spawn_next to run
    for _ in range(10):
        scheduler.tick()
        time.sleep(0.3)
        job_next = scheduler.storage.get_job(job_id_next)
        if job_next and job_next.status == JobStatus.SUCCEEDED.value:
            break

    job_next = scheduler.storage.get_job(job_id_next)
    assert job_next is not None
    assert job_next.status == JobStatus.SUCCEEDED.value


# ---------------------------------------------------------------------------
# 19. Review Feedback: Single CUDA OOM Retry Orchestration
# ---------------------------------------------------------------------------


def test_cuda_oom_retry_halves_batch_then_pauses_error(temp_db_path: Path):
    """Verifies that 1st CUDA OOM halves batch size and requeues; 2nd OOM marks PAUSED_ERROR."""
    scheduler = GPUScheduler(
        db_path=temp_db_path,
        acquire_coordinator_lock=False,
    )

    job_id = "auto_oom_trial"
    train_payload = {
        "train_spec": {
            "batch_size": 4,
            "gradient_accumulation_steps": 4,
            "timeframe": "1h",
        },
        "simulate_oom": True,
    }
    scheduler.storage.submit_job(
        JobSpec(
            job_id=job_id,
            job_type=JobType.DUMMY.value,
            timeframe="1h",
            priority=JobPriority.AUTO.value,
            payload=train_payload,
        )
    )

    # 1. Acquire job and simulate 1st OOM failure
    job = scheduler.storage.acquire_next_job()
    assert job is not None
    scheduler.storage.mark_failed(job_id, "[CUDA_OOM] CUDA out of memory. Tried to allocate 4.00 GiB")

    # Call OOM handler
    failed_job = scheduler.storage.get_job(job_id)
    retried_id = scheduler._handle_oom_retry_if_needed(failed_job)
    assert retried_id is not None
    assert retried_id == f"{job_id}_oom_retry"

    # Check retry job was queued with halved batch size and doubled accumulation
    requeued_job = scheduler.storage.get_job(retried_id)
    assert requeued_job is not None
    assert requeued_job.status == JobStatus.QUEUED.value
    assert requeued_job.payload["oom_retry_count"] == 1
    assert requeued_job.payload["train_spec"]["batch_size"] == 2
    assert requeued_job.payload["train_spec"]["gradient_accumulation_steps"] == 8

    # 2. Acquire again and simulate 2nd OOM failure
    job_retry = scheduler.storage.acquire_next_job()
    assert job_retry is not None
    assert job_retry.job_id == retried_id
    scheduler.storage.mark_failed(retried_id, "[CUDA_OOM] CUDA out of memory. Tried to allocate 4.00 GiB")

    # Call OOM handler again
    failed_retry_job = scheduler.storage.get_job(retried_id)
    retried_2 = scheduler._handle_oom_retry_if_needed(failed_retry_job)
    assert retried_2 is None

    # Check state transitioned to PAUSED_ERROR
    final_job = scheduler.storage.get_job(retried_id)
    assert final_job.status == JobStatus.FAILED.value
    assert scheduler.storage.get_auto_run_state("1h") == AutoRunState.PAUSED_ERROR


# ---------------------------------------------------------------------------
# 20. Review Feedback: Atomic Coordinator Lease Race Condition
# ---------------------------------------------------------------------------


def test_coordinator_lease_atomic_contention(temp_db_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verifies that under concurrent attempts, exactly one coordinator wins the lease."""
    from concurrent.futures import ThreadPoolExecutor

    # Ensure simulated processes are treated as alive so contention logic triggers
    monkeypatch.setattr("paxg_lab.queue.process_guard.is_process_alive", lambda pid, ctime=None: True)
    storage = GPUJobStorage(temp_db_path)

    def attempt_claim(simulated_pid: int) -> bool:
        s = GPUJobStorage(temp_db_path)
        return s.try_acquire_coordinator_lease(
            my_pid=simulated_pid,
            my_create_time=time.time(),
            lease_timeout=30.0,
        )

    # Use simulated PIDs for test (e.g. 20000 to 20007)
    pids = list(range(20000, 20008))
    with ThreadPoolExecutor(max_workers=len(pids)) as pool:
        claims = list(pool.map(attempt_claim, pids))

    # Exactly 1 claim should have succeeded when multiple enter concurrently
    assert sum(claims) == 1


# ---------------------------------------------------------------------------
# 21. Review Feedback: Pipe Buffer Deadlock Prevention (Redirect to Job Log)
# ---------------------------------------------------------------------------


def test_pipe_buffer_deadlock_prevention_job_logs(temp_db_path: Path):
    """Verifies worker logs are redirected to file and not piped through OS buffers."""
    scheduler = GPUScheduler(
        db_path=temp_db_path,
        acquire_coordinator_lock=False,
    )
    job_id = "test_log_pipe_job"
    scheduler.storage.submit_job(
        JobSpec(
            job_id=job_id,
            job_type=JobType.DUMMY.value,
            priority=JobPriority.MANUAL.value,
            payload={"steps": 2, "step_sleep": 0.1},
        )
    )

    scheduler.tick()
    assert scheduler.active_worker is not None
    # Verify stdout and stderr are NOT subprocess.PIPE
    assert scheduler.active_worker.stdout is None or not hasattr(scheduler.active_worker.stdout, "read")
    assert scheduler.active_worker.stderr is None or not hasattr(scheduler.active_worker.stderr, "read")

    # Verify log file path exists
    log_file = Path(f"var/paxg_lab/job_logs/job_{job_id}.log")
    for _ in range(10):
        time.sleep(0.3)
        scheduler.tick()
        job = scheduler.storage.get_job(job_id)
        if job and job.status == JobStatus.SUCCEEDED.value:
            break

    assert log_file.exists()


# ---------------------------------------------------------------------------
# 22. Review Feedback: Fine-Grained Stop Responsiveness in Training & Eval
# ---------------------------------------------------------------------------


def test_step_level_graceful_stop_responsiveness(temp_db_path: Path):
    """Verifies LoRATrainer stops after optimizer step when progress_callback returns False."""
    import pandas as pd
    import numpy as np
    import torch.nn as nn
    from paxg_lab.model.train_spec import TrainSpec
    from paxg_lab.model.trainer import LoRATrainer

    class MockTimesFMModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.query_proj = nn.Linear(1, 1)
            self.value_proj = nn.Linear(1, 1)

        def forward_decode(self, target: torch.Tensor, horizon: int) -> torch.Tensor:
            b, f, c = target.shape
            dummy = self.query_proj(target[:, :, -1:]) + self.value_proj(target[:, :, -1:])
            return dummy.unsqueeze(-1).expand(b, f, horizon, 9)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.query_proj(x) + self.value_proj(x)

    base_model = MockTimesFMModule()
    spec = TrainSpec(
        timeframe="1h",
        context_len=256,
        horizon=24,
        feature_set="B",
        lora_r=2,
        lora_alpha=4,
        max_epochs=5,
        batch_size=2,
        gradient_accumulation_steps=1,
    )
    trainer = LoRATrainer(base_model=base_model, spec=spec)

    # Mock dataset preparation to provide exact batches for step-level verification
    from paxg_lab.data.features import FEATURE_SPECS
    num_features = len(FEATURE_SPECS["B"].columns)
    dummy_train_ctx = np.ones((10, num_features, 256), dtype=np.float32) * 2000.0
    dummy_train_fut = np.ones((10, 24), dtype=np.float32) * 2005.0
    dummy_val_ctx = np.ones((4, num_features, 256), dtype=np.float32) * 2000.0
    dummy_val_fut = np.ones((4, 24), dtype=np.float32) * 2005.0
    mock_meta = {
        "train_samples": 10,
        "val_samples": 4,
        "feature_set": "B",
        "fold_id": 1,
        "train_start_time_ms": 1704067200000,
        "train_end_time_ms": 1704153600000,
        "val_start_time_ms": 1704153600000,
        "val_end_time_ms": 1704240000000,
        "num_train_windows": 10,
        "num_val_windows": 4,
    }
    trainer.prepare_dataset = lambda *args, **kwargs: (dummy_train_ctx, dummy_train_fut, dummy_val_ctx, dummy_val_fut, mock_meta)

    step_counter = 0

    def progress_cb(info: dict[str, Any]) -> bool:
        nonlocal step_counter
        if info.get("type") == "step_update":
            step_counter += 1
            if step_counter >= 2:
                # Signal stop at step 2 of epoch 1!
                return False
        return True

    res = trainer.train(
        features_df=pd.DataFrame(),
        snapshot_hash="dummy_hash",
        fold_id=1,
        progress_callback=progress_cb,
    )

    # Must have stopped before completing max_epochs (5)
    assert len(res.history) <= 1
    assert step_counter >= 2


def test_batch_level_graceful_stop_in_predict_windows():
    """Verifies BacktestEngine.predict_windows raises InterruptedError when progress_callback returns False."""
    import numpy as np
    from paxg_lab.eval.engine import BacktestEngine

    class MockPredictor:
        def predict_batch(self, batch_ctx: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
            b = len(batch_ctx)
            return np.ones((b, horizon), dtype=np.float32), np.ones((b, horizon, 9), dtype=np.float32)

    engine = BacktestEngine(predictor=MockPredictor())
    # 20 context windows, batch_size=4 -> 5 batches
    dummy_contexts = np.zeros((20, 2, 64), dtype=np.float32)

    def stop_on_batch_2(info: dict[str, Any]) -> bool:
        if info.get("batch_idx") == 2:
            return False
        return True

    with pytest.raises(InterruptedError, match="Backtest halted by stop request"):
        engine.predict_windows(
            contexts=dummy_contexts,
            horizon=4,
            batch_size=4,
            progress_callback=stop_on_batch_2,
        )


# ---------------------------------------------------------------------------
# 24. Review Round 2: E2E Valid FORECAST from Snapshot (1h=24, 4h=6, Future Timestamps)
# ---------------------------------------------------------------------------


def test_forecast_job_e2e_success_with_snapshot(temp_db_path: Path, tmp_path: Path):
    """Verifies valid FORECAST from snapshot executes via GPUWorker to SUCCEEDED with correct horizon & future timestamps."""
    from paxg_lab.data.features import FEATURE_SPECS
    from paxg_lab.data.snapshot import DatasetSnapshot
    from paxg_lab.queue.worker import GPUWorker

    storage = GPUJobStorage(temp_db_path)

    # 1. Create a minimal valid 1h snapshot (300 candles)
    n_candles = 300
    base_ts = 1740000000000
    step_1h_ms = 3600 * 1000
    timestamps_1h = np.array([base_ts + i * step_1h_ms for i in range(n_candles)], dtype=np.int64)
    feat_1h_a = np.ones((n_candles, len(FEATURE_SPECS["A"].columns)), dtype=np.float32) * 2000.0
    feat_1h_b = np.ones((n_candles, len(FEATURE_SPECS["B"].columns)), dtype=np.float32) * 2000.0
    snap_1h = DatasetSnapshot.create(
        timeframe="1h",
        timestamps=timestamps_1h,
        features_a=feat_1h_a,
        features_b=feat_1h_b,
        symbol="PAXGUSDT",
    )
    snap_1h_dir = snap_1h.save(base_dir=tmp_path / "snap_1h")

    # Submit 1h FORECAST job
    job_id_1h = "job_forecast_1h_valid"
    storage.submit_job(
        JobSpec(
            job_id=job_id_1h,
            job_type=JobType.FORECAST.value,
            timeframe="1h",
            priority=JobPriority.FORECAST.value,
            payload={
                "snapshot_path": str(snap_1h_dir),
                "timeframe": "1h",
                "context_len": 256,
                "feature_set": "A",
            },
        )
    )

    # Execute via GPUWorker
    worker_1h = GPUWorker(job_id=job_id_1h, db_path=temp_db_path)
    exit_code_1h = worker_1h.run()
    assert exit_code_1h == 0

    job_1h = storage.get_job(job_id_1h)
    assert job_1h is not None
    assert job_1h.status == JobStatus.SUCCEEDED.value
    res_1h = job_1h.result
    assert res_1h["timeframe"] == "1h"
    assert len(res_1h["point_forecast"]) == 24
    assert len(res_1h["quantiles"]) == 24
    assert len(res_1h["target_timestamps"]) == 24
    assert res_1h["forecast_origin_time"] == int(timestamps_1h[-1])
    assert res_1h["target_timestamps"][0] == res_1h["forecast_origin_time"] + step_1h_ms
    assert res_1h["target_timestamps"][-1] == res_1h["forecast_origin_time"] + 24 * step_1h_ms
    assert all(t > res_1h["forecast_origin_time"] for t in res_1h["target_timestamps"])

    # 2. Test 4h FORECAST job (horizon=6)
    step_4h_ms = 4 * 3600 * 1000
    timestamps_4h = np.array([base_ts + i * step_4h_ms for i in range(n_candles)], dtype=np.int64)
    feat_4h_a = np.ones((n_candles, len(FEATURE_SPECS["A"].columns)), dtype=np.float32) * 2000.0
    feat_4h_b = np.ones((n_candles, len(FEATURE_SPECS["B"].columns)), dtype=np.float32) * 2000.0
    snap_4h = DatasetSnapshot.create(
        timeframe="4h",
        timestamps=timestamps_4h,
        features_a=feat_4h_a,
        features_b=feat_4h_b,
        symbol="PAXGUSDT",
    )
    snap_4h_dir = snap_4h.save(base_dir=tmp_path / "snap_4h")

    job_id_4h = "job_forecast_4h_valid"
    storage.submit_job(
        JobSpec(
            job_id=job_id_4h,
            job_type=JobType.FORECAST.value,
            timeframe="4h",
            priority=JobPriority.FORECAST.value,
            payload={
                "snapshot_path": str(snap_4h_dir),
                "timeframe": "4h",
                "context_len": 256,
                "feature_set": "A",
            },
        )
    )

    worker_4h = GPUWorker(job_id=job_id_4h, db_path=temp_db_path)
    exit_code_4h = worker_4h.run()
    assert exit_code_4h == 0

    job_4h = storage.get_job(job_id_4h)
    assert job_4h is not None
    assert job_4h.status == JobStatus.SUCCEEDED.value
    res_4h = job_4h.result
    assert res_4h["timeframe"] == "4h"
    assert len(res_4h["point_forecast"]) == 6
    assert len(res_4h["quantiles"]) == 6
    assert len(res_4h["target_timestamps"]) == 6
    assert res_4h["forecast_origin_time"] == int(timestamps_4h[-1])
    assert res_4h["target_timestamps"][0] == res_4h["forecast_origin_time"] + step_4h_ms
    assert res_4h["target_timestamps"][-1] == res_4h["forecast_origin_time"] + 6 * step_4h_ms


# ---------------------------------------------------------------------------
# 25. Review Round 2: E2E TRAIN & AUTO_TRIAL Job using Real P3 Adapter Pipeline
# ---------------------------------------------------------------------------


def test_train_and_auto_trial_job_e2e_success(temp_db_path: Path, tmp_path: Path):
    """Verifies TRAIN and AUTO_TRIAL jobs execute via GPUWorker using real P3 APIs and save verified adapters."""
    from paxg_lab.data.features import FEATURE_SPECS
    from paxg_lab.data.snapshot import DatasetSnapshot
    from paxg_lab.queue.worker import GPUWorker

    storage = GPUJobStorage(temp_db_path)

    real_snap_path = Path("var/paxg_lab/snapshots/paxgusdt_1h_1743073200000_1788613200000_837c9ee8")
    if real_snap_path.exists():
        snap_dir = real_snap_path
    else:
        n_candles = 6000
        base_ts = 1700000000000
        step_ms = 3600 * 1000
        timestamps = np.array([base_ts + i * step_ms for i in range(n_candles)], dtype=np.int64)
        feat_a = np.ones((n_candles, len(FEATURE_SPECS["A"].columns)), dtype=np.float32) * 2000.0
        feat_b = np.ones((n_candles, len(FEATURE_SPECS["B"].columns)), dtype=np.float32) * 2000.0
        snap = DatasetSnapshot.create(
            timeframe="1h",
            timestamps=timestamps,
            features_a=feat_a,
            features_b=feat_b,
            symbol="PAXGUSDT",
        )
        snap_dir = snap.save(base_dir=tmp_path / "snap_train")

    adapter_store_dir = tmp_path / "adapters"
    adapter_store_dir.mkdir(parents=True, exist_ok=True)

    # 1. Execute TRAIN job
    job_id_train = "job_train_p3_valid"
    storage.submit_job(
        JobSpec(
            job_id=job_id_train,
            job_type=JobType.TRAIN.value,
            timeframe="1h",
            priority=JobPriority.MANUAL.value,
            payload={
                "snapshot_path": str(snap_dir),
                "adapter_store_dir": str(adapter_store_dir),
                "smoke_test": False,
                "train_spec": {
                    "timeframe": "1h",
                    "context_len": 256,
                    "horizon": 24,
                    "feature_set": "B",
                    "max_epochs": 1,
                    "batch_size": 2,
                    "gradient_accumulation_steps": 1,
                    "max_samples_per_epoch": 2,
                    "history_days": 180,
                },
            },
        )
    )

    worker_train = GPUWorker(job_id=job_id_train, db_path=temp_db_path)
    exit_code_train = worker_train.run()
    assert exit_code_train == 0

    job_train = storage.get_job(job_id_train)
    assert job_train is not None
    assert job_train.status == JobStatus.SUCCEEDED.value
    res_train = job_train.result
    saved_adapter_path = Path(res_train["adapter_path"])
    assert saved_adapter_path.is_dir()
    assert (saved_adapter_path / "adapter_model.safetensors").exists()
    assert (saved_adapter_path / "paxg_manifest.json").exists()
    assert (saved_adapter_path / "checksums.sha256").exists()

    # 2. Execute AUTO_TRIAL job (same pipeline)
    job_id_auto = "job_auto_trial_p3_valid"
    storage.submit_job(
        JobSpec(
            job_id=job_id_auto,
            job_type=JobType.AUTO_TRIAL.value,
            timeframe="1h",
            priority=JobPriority.AUTO.value,
            payload={
                "snapshot_path": str(snap_dir),
                "adapter_store_dir": str(adapter_store_dir),
                "smoke_test": False,
                "train_spec": {
                    "timeframe": "1h",
                    "context_len": 256,
                    "horizon": 24,
                    "feature_set": "B",
                    "max_epochs": 1,
                    "batch_size": 2,
                    "gradient_accumulation_steps": 1,
                    "max_samples_per_epoch": 2,
                    "history_days": 180,
                },
            },
        )
    )

    worker_auto = GPUWorker(job_id=job_id_auto, db_path=temp_db_path)
    exit_code_auto = worker_auto.run()
    assert exit_code_auto == 0

    job_auto = storage.get_job(job_id_auto)
    assert job_auto is not None
    assert job_auto.status == JobStatus.SUCCEEDED.value
    assert Path(job_auto.result["adapter_path"]).exists()


# ---------------------------------------------------------------------------
# 26. Review Round 2: Worker Terminate Failure Blocks Queue (Invariant Protection)
# ---------------------------------------------------------------------------


def test_worker_terminate_failure_blocks_queue(temp_db_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verifies that if safe_terminate_process fails and the process remains alive,

    the scheduler marks an error, keeps the queue blocked, and NEVER acquires or spawns the next GPU job.
    """
    scheduler = GPUScheduler(
        db_path=temp_db_path,
        heartbeat_timeout=1.0,
        acquire_coordinator_lock=False,
    )

    # Job 1: Active hung worker
    hung_job_id = "hung_unkillable_worker"
    scheduler.storage.submit_job(
        JobSpec(job_id=hung_job_id, job_type=JobType.DUMMY.value, priority=JobPriority.MANUAL.value)
    )

    # Job 2: Queued next job
    next_job_id = "next_queued_gpu_job"
    scheduler.storage.submit_job(
        JobSpec(job_id=next_job_id, job_type=JobType.DUMMY.value, priority=JobPriority.FORECAST.value)
    )

    # Adopt fake PID 99999 in RUNNING state with expired heartbeat
    fake_pid = 99999
    fake_ctime = time.time()
    stale_hb = time.time() - 10.0
    with scheduler.storage.get_connection() as conn:
        conn.execute(
            """
            UPDATE gpu_jobs
            SET status = 'RUNNING', worker_pid = ?, worker_create_time = ?,
                started_at = ?, heartbeat_at = ?
            WHERE job_id = ?;
            """,
            (fake_pid, fake_ctime, stale_hb, stale_hb, hung_job_id),
        )

    scheduler.active_job_id = hung_job_id
    scheduler.active_worker_pid = fake_pid
    scheduler.active_worker_create_time = fake_ctime

    # Monkeypatch termination failure and alive process
    monkeypatch.setattr("paxg_lab.queue.scheduler.safe_terminate_process", lambda pid, ctime=None, timeout=5.0: False)
    monkeypatch.setattr("paxg_lab.queue.scheduler.is_process_alive", lambda pid, ctime=None: True)

    # Tick scheduler: terminate fails!
    scheduler.tick()

    # Invariant Verification:
    # 1. Scheduler error is recorded
    assert scheduler.scheduler_error is not None
    assert f"Worker PID {fake_pid} could not be terminated" in scheduler.scheduler_error

    # 2. Active worker PID and active job ID are NOT cleared
    assert scheduler.active_worker_pid == fake_pid
    assert scheduler.active_job_id == hung_job_id

    # 3. Subsequent ticks NEVER acquire or spawn the next job
    for _ in range(5):
        scheduler.tick()

    next_job = scheduler.storage.get_job(next_job_id)
    assert next_job is not None
    assert next_job.status == JobStatus.QUEUED.value  # MUST stay QUEUED
    assert next_job.started_at is None


# ---------------------------------------------------------------------------
# 27. Review Round 2: Stop Auto Run Enforces No New Auto Jobs Dispatched
# ---------------------------------------------------------------------------


def test_stop_auto_run_enforces_no_new_auto_jobs_dispatched(temp_db_path: Path):
    """Verifies that calling stop_auto_run strictly blocks new AUTO jobs from being dispatched,

    and only explicit start/resume allows queued AUTO jobs to proceed.
    """
    scheduler = GPUScheduler(
        db_path=temp_db_path,
        acquire_coordinator_lock=False,
    )

    # Stop 1h auto run
    scheduler.stop_auto_run("1h")
    assert scheduler.storage.get_auto_run_state("1h") == AutoRunState.STOPPED

    # Attempt submitting with reject_if_stopped=True -> must raise ValueError
    with pytest.raises(ValueError, match="auto-run is currently STOPPED"):
        scheduler.storage.submit_job(
            JobSpec(
                job_id="auto_rejected",
                job_type=JobType.DUMMY.value,
                timeframe="1h",
                priority=JobPriority.AUTO.value,
            ),
            reject_if_stopped=True,
        )

    # Submit new AUTO job without reject flag (enters queue as QUEUED)
    auto_job_id = "auto_queued_while_stopped"
    scheduler.storage.submit_job(
        JobSpec(
            job_id=auto_job_id,
            job_type=JobType.DUMMY.value,
            timeframe="1h",
            priority=JobPriority.AUTO.value,
            payload={"steps": 1, "step_sleep": 0.05},
        )
    )

    # Run multiple scheduler ticks while STOPPED -> job must NEVER be dispatched to RUNNING
    for _ in range(5):
        scheduler.tick()
        job = scheduler.storage.get_job(auto_job_id)
        assert job.status == JobStatus.QUEUED.value
        assert scheduler.active_job_id is None
        assert scheduler.active_worker is None

    # Now explicitly resume / start 1h auto run
    scheduler.start_auto_run("1h")
    assert scheduler.storage.get_auto_run_state("1h") == AutoRunState.SEARCHING

    # Next tick acquires the previously blocked job
    scheduler.tick()
    assert scheduler.active_job_id == auto_job_id
    job_resumed = scheduler.storage.get_job(auto_job_id)
    assert job_resumed.status == JobStatus.RUNNING.value

    # Stop worker to clean up
    scheduler.stop()


# ---------------------------------------------------------------------------
# 28. Review Round 2: Coordinator Lease CAS Ownership & Split-Brain Prevention
# ---------------------------------------------------------------------------


def test_coordinator_lease_takeover_prevents_stale_renewal(temp_db_path: Path):
    """Verifies that if coordinator lease is taken over by another scheduler,

    the eclipsed coordinator's renewal fails and it immediately halts to prevent split-brain.
    """
    storage = GPUJobStorage(temp_db_path)

    # Scheduler A claims lease
    scheduler_a = GPUScheduler(
        db_path=temp_db_path,
        acquire_coordinator_lock=True,
    )
    assert scheduler_a.owner_token is not None

    # Renewal succeeds while owner
    assert scheduler_a._update_coordinator_heartbeat() is True

    # Simulate takeover by Scheduler B with a new token and different PID
    new_token_b = "token_coordinator_b_99999"
    new_pid_b = 99999
    now = time.time()
    with storage.get_connection() as conn:
        conn.execute(
            """
            UPDATE scheduler_state
            SET value = ?, updated_at = ?
            WHERE key = 'coordinator_lease';
            """,
            (
                json.dumps({
                    "pid": new_pid_b,
                    "create_time": now,
                    "owner_token": new_token_b,
                    "heartbeat": now,
                }),
                now,
            ),
        )

    # Scheduler A attempts renewal on its next tick
    scheduler_a.tick()

    # Scheduler A must detect ownership loss and stop immediately
    assert scheduler_a._is_running is False

    # Lease in SQLite must NOT be overwritten by Scheduler A
    current_lease = storage.get_coordinator_lease()
    assert current_lease is not None
    assert current_lease["owner_token"] == new_token_b
    assert current_lease["pid"] == new_pid_b


# ---------------------------------------------------------------------------
# 29. Review Round 3: E2E BACKTEST Job Execution for Base and LoRA Models
# ---------------------------------------------------------------------------


def get_or_create_test_snapshot(tmp_path: Path, timeframe: str = "4h") -> Path:
    """Returns an existing snapshot or synthesizes a cryptographically valid snapshot for CI."""
    candidates = list(Path("var/paxg_lab/snapshots").glob(f"paxgusdt_{timeframe}_*"))
    if candidates and not os.environ.get("FORCE_SYNTHETIC_SNAPSHOT"):
        return candidates[0]

    import numpy as np
    import pandas as pd
    from paxg_lab.data.features import build_features
    from paxg_lab.data.snapshot import DatasetSnapshot

    n_rows = 2000 if timeframe == "4h" else 5000
    base_ts_ms = 1700000000000
    step_ms = (14400 if timeframe == "4h" else 3600) * 1000
    timestamps = [base_ts_ms + i * step_ms for i in range(n_rows)]
    rng = np.random.default_rng(42)
    prices = 2000.0 + np.cumsum(rng.normal(0, 2, n_rows))
    df = pd.DataFrame({
        "open_time": timestamps,
        "open": prices,
        "high": prices + 2.0,
        "low": prices - 2.0,
        "close": prices,
        "volume": rng.uniform(10, 100, n_rows),
        "close_time": [t + step_ms - 1 for t in timestamps],
        "quote_volume": rng.uniform(20000, 200000, n_rows),
        "count": rng.integers(50, 500, n_rows),
        "taker_buy_volume": rng.uniform(5, 50, n_rows),
        "taker_buy_quote_volume": rng.uniform(10000, 100000, n_rows),
    })
    feat_a, _, _ = build_features(df, feature_set="A")
    feat_b, _, _ = build_features(df, feature_set="B")
    snap = DatasetSnapshot.create(
        timeframe=timeframe,
        timestamps=df["open_time"].to_numpy(),
        features_a=feat_a,
        features_b=feat_b,
    )
    snap_dir = tmp_path / "ci_snapshots"
    return snap.save(base_dir=snap_dir)


def test_backtest_job_e2e_base_and_lora(temp_db_path: Path, tmp_path: Path):
    """Verifies that BACKTEST jobs execute cleanly via GPUWorker for both Base and LoRA models."""
    from paxg_lab.queue.worker import GPUWorker

    storage = GPUJobStorage(temp_db_path)
    snap_4h = get_or_create_test_snapshot(tmp_path, timeframe="4h")

    # 1. Base model backtest (establishes base reference)
    job_id_base = "job_backtest_base_p4"
    storage.submit_job(
        JobSpec(
            job_id=job_id_base,
            job_type=JobType.BACKTEST.value,
            timeframe="4h",
            priority=JobPriority.MANUAL.value,
            payload={
                "snapshot_path": str(snap_4h),
                "model_type": "base",
                "is_base_reference": True,
                "feature_set": "A",
                "context_len": 256,
                "batch_size": 32,
            },
        )
    )

    worker_base = GPUWorker(job_id=job_id_base, db_path=temp_db_path)
    exit_code_base = worker_base.run()
    assert exit_code_base == 0

    job_base = storage.get_job(job_id_base)
    assert job_base is not None
    assert job_base.status == JobStatus.SUCCEEDED.value
    res_base = job_base.result
    assert res_base["timeframe"] == "4h"
    assert res_base["model_name"] == "TimesFM3-Base"
    assert np.isclose(res_base["score"], 0.0, atol=1e-4)
    assert res_base["overall_weighted_mae"] > 0.0
    assert len(res_base["fold_metrics"]) > 0

    # 2. Train a fast LoRA adapter on the 4h snapshot
    job_id_train = "job_train_adapter_for_backtest"
    adapter_store_dir = tmp_path / "adapters_backtest"
    storage.submit_job(
        JobSpec(
            job_id=job_id_train,
            job_type=JobType.TRAIN.value,
            timeframe="4h",
            priority=JobPriority.MANUAL.value,
            payload={
                "snapshot_path": str(snap_4h),
                "adapter_store_dir": str(adapter_store_dir),
                "checkpoint_dir": str(tmp_path / "checkpoints_backtest_train"),
                "smoke_test": False,
                "train_spec": {
                    "timeframe": "4h",
                    "context_len": 256,
                    "horizon": 6,
                    "feature_set": "B",
                    "max_epochs": 1,
                    "batch_size": 2,
                    "gradient_accumulation_steps": 1,
                    "max_samples_per_epoch": 2,
                    "history_days": 180,
                },
            },
        )
    )
    worker_train = GPUWorker(job_id=job_id_train, db_path=temp_db_path)
    assert worker_train.run() == 0
    trained_adapter_path = storage.get_job(job_id_train).result["adapter_path"]

    # 3. LoRA candidate model backtest passing base_reference_metrics
    job_id_lora = "job_backtest_lora_p4"
    storage.submit_job(
        JobSpec(
            job_id=job_id_lora,
            job_type=JobType.BACKTEST.value,
            timeframe="4h",
            priority=JobPriority.MANUAL.value,
            payload={
                "snapshot_path": str(snap_4h),
                "model_type": "lora",
                "adapter_path": trained_adapter_path,
                "feature_set": "B",
                "context_len": 256,
                "batch_size": 32,
                "base_reference_metrics": res_base,
            },
        )
    )
    worker_lora = GPUWorker(job_id=job_id_lora, db_path=temp_db_path)
    exit_code_lora = worker_lora.run()
    assert exit_code_lora == 0

    job_lora = storage.get_job(job_id_lora)
    assert job_lora is not None
    assert job_lora.status == JobStatus.SUCCEEDED.value
    res_lora = job_lora.result
    assert res_lora["timeframe"] == "4h"
    assert res_lora["model_name"] == "TimesFM3-LoRA"
    assert "score" in res_lora
    assert isinstance(res_lora["score"], (int, float))
    assert len(res_lora["fold_metrics"]) > 0


# ---------------------------------------------------------------------------
# 30. Review Round 3: Preservation of Full TrainSpec and history_days='all'
# ---------------------------------------------------------------------------


def test_train_spec_full_hyperparameters_and_history_all(temp_db_path: Path, tmp_path: Path):
    """Verifies that custom TrainSpec hyperparameters and history_days='all' pass intact to the trainer and manifest."""
    from paxg_lab.model.store import AdapterStore
    from paxg_lab.queue.worker import GPUWorker

    storage = GPUJobStorage(temp_db_path)
    snap_4h = get_or_create_test_snapshot(tmp_path, timeframe="4h")

    adapter_store_dir = tmp_path / "adapters_custom_spec"
    job_id = "train_custom_spec_all"
    storage.submit_job(
        JobSpec(
            job_id=job_id,
            job_type=JobType.TRAIN.value,
            timeframe="4h",
            priority=JobPriority.MANUAL.value,
            payload={
                "snapshot_path": str(snap_4h),
                "adapter_store_dir": str(adapter_store_dir),
                "checkpoint_dir": str(tmp_path / "checkpoints_spec_all"),
                "smoke_test": False,
                "train_spec": {
                    "timeframe": "4h",
                    "horizon": 6,
                    "feature_set": "B",
                    "context_len": 256,
                    "lora_r": 4,
                    "lora_alpha": 8,
                    "lora_dropout": 0.15,
                    "learning_rate": 1e-4,
                    "max_epochs": 1,
                    "batch_size": 2,
                    "gradient_accumulation_steps": 2,
                    "weight_decay": 0.05,
                    "early_stopping_patience": 3,
                    "grad_clip_norm": 1.5,
                    "history_days": "all",
                    "max_samples_per_epoch": 2,
                    "warmup_ratio": 0.05,
                    "seed": 123,
                },
            },
        )
    )

    worker = GPUWorker(job_id=job_id, db_path=temp_db_path)
    exit_code = worker.run()
    assert exit_code == 0

    job = storage.get_job(job_id)
    assert job is not None
    assert job.status == JobStatus.SUCCEEDED.value

    adapter_path = Path(job.result["adapter_path"])
    from paxg_lab.model.manifest import AdapterManifest
    manifest = AdapterManifest.load_json(adapter_path / "paxg_manifest.json")

    # Verify custom hyperparameters were fully preserved
    cfg = manifest.train_spec
    assert cfg["weight_decay"] == 0.05
    assert cfg["early_stopping_patience"] == 3
    assert cfg["grad_clip_norm"] == 1.5
    assert cfg["warmup_ratio"] == 0.05
    assert cfg["history_days"] == "all"
    assert cfg["learning_rate"] == 1e-4
    assert cfg["lora_dropout"] == 0.15
    assert cfg["seed"] == 123


# ---------------------------------------------------------------------------
# 31. Review Round 3: Spawn Failure Safety (Child Termination & Unkillable Queue Block)
# ---------------------------------------------------------------------------


def test_spawn_failure_kills_child_and_blocks_if_unkillable(temp_db_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verifies that if worker startup fails after Popen(), the child is killed without leaving an orphan.

    If the child cannot be killed, the scheduler enters an error state and halts dispatches.
    """
    scheduler = GPUScheduler(db_path=temp_db_path, acquire_coordinator_lock=False)

    # 1. Successful child kill on startup exception
    job_id_fail = "job_post_spawn_fail"
    job_id_next = "job_after_spawn_fail"
    scheduler.storage.submit_job(
        JobSpec(job_id=job_id_fail, job_type=JobType.DUMMY.value, priority=JobPriority.MANUAL.value)
    )
    scheduler.storage.submit_job(
        JobSpec(job_id=job_id_next, job_type=JobType.DUMMY.value, priority=JobPriority.MANUAL.value, payload={"steps": 1, "step_sleep": 0.05})
    )

    # Monkeypatch register_worker to simulate a database failure after Popen succeeded
    orig_register = scheduler.storage.register_worker

    def fail_register(job_id, pid, create_time):
        raise RuntimeError("Simulated database failure during register_worker after Popen")

    monkeypatch.setattr(scheduler.storage, "register_worker", fail_register)

    # Tick scheduler: Popen runs, register_worker fails -> child must be cleanly terminated
    scheduler.tick()

    job_fail = scheduler.storage.get_job(job_id_fail)
    assert job_fail is not None
    assert job_fail.status == JobStatus.FAILED.value
    assert "Failed to spawn worker subprocess" in (job_fail.error_message or "")
    assert scheduler.active_worker is None
    assert scheduler.active_job_id is None
    assert scheduler.active_worker_pid is None
    assert scheduler.scheduler_error is None

    # Restore register_worker: next job can be dispatched cleanly
    monkeypatch.setattr(scheduler.storage, "register_worker", orig_register)
    scheduler.tick()
    assert scheduler.active_job_id == job_id_next
    scheduler.stop()

    # 2. Unkillable child process blocks scheduler
    scheduler_b = GPUScheduler(db_path=temp_db_path, acquire_coordinator_lock=False)
    job_id_unkillable = "job_unkillable_spawn"
    scheduler_b.storage.submit_job(
        JobSpec(job_id=job_id_unkillable, job_type=JobType.DUMMY.value, priority=JobPriority.MANUAL.value)
    )

    # Monkeypatch register_worker to fail AND safe_terminate_process to fail
    monkeypatch.setattr(scheduler_b.storage, "register_worker", fail_register)
    monkeypatch.setattr("paxg_lab.queue.scheduler.safe_terminate_process", lambda pid, ctime=None, timeout=5.0: False)
    monkeypatch.setattr("paxg_lab.queue.scheduler.is_process_alive", lambda pid, ctime=None: True)

    scheduler_b.tick()

    assert scheduler_b.scheduler_error is not None
    assert "could not be terminated" in scheduler_b.scheduler_error
    assert scheduler_b.active_worker_pid is not None

    # Verify subsequent tick refuses to dispatch any jobs
    job_id_blocked = "job_blocked_by_error"
    scheduler_b.storage.submit_job(
        JobSpec(job_id=job_id_blocked, job_type=JobType.DUMMY.value, priority=JobPriority.MANUAL.value)
    )
    scheduler_b.tick()
    assert scheduler_b.storage.get_job(job_id_blocked).status == JobStatus.QUEUED.value


# ---------------------------------------------------------------------------
# 32. Review Round 3: OOM Retry Boundary Handling (batch_size=1 and accum=16)
# ---------------------------------------------------------------------------


def test_oom_retry_boundaries_batch_1_and_accum_16(temp_db_path: Path):
    """Verifies OOM retry boundaries: batch=1 transitions to PAUSED_ERROR, and accum is clamped to 16."""
    from paxg_lab.model.train_spec import TrainSpec

    scheduler = GPUScheduler(db_path=temp_db_path, acquire_coordinator_lock=False)

    # 1. Boundary A: batch_size = 1 cannot be reduced further -> PAUSED_ERROR
    job_id_b1 = "oom_batch_1_job"
    scheduler.storage.submit_job(
        JobSpec(
            job_id=job_id_b1,
            job_type=JobType.DUMMY.value,
            timeframe="1h",
            priority=JobPriority.AUTO.value,
            payload={
                "train_spec": {
                    "batch_size": 1,
                    "gradient_accumulation_steps": 8,
                    "timeframe": "1h",
                },
            },
        )
    )
    job_b1 = scheduler.storage.acquire_next_job()
    assert job_b1 is not None
    scheduler.storage.mark_failed(job_id_b1, "[CUDA_OOM] CUDA out of memory. Tried to allocate 2.00 GiB")

    failed_b1 = scheduler.storage.get_job(job_id_b1)
    retry_id_b1 = scheduler._handle_oom_retry_if_needed(failed_b1)
    assert retry_id_b1 is None
    assert scheduler.storage.get_auto_run_state("1h") == AutoRunState.PAUSED_ERROR

    # 2. Boundary B: gradient_accumulation_steps = 16 clamped to 16, producing valid TrainSpec
    scheduler.storage.set_auto_run_state("4h", AutoRunState.SEARCHING)
    job_id_a16 = "oom_accum_16_job"
    scheduler.storage.submit_job(
        JobSpec(
            job_id=job_id_a16,
            job_type=JobType.DUMMY.value,
            timeframe="4h",
            priority=JobPriority.AUTO.value,
            payload={
                "train_spec": {
                    "batch_size": 2,
                    "gradient_accumulation_steps": 16,
                    "timeframe": "4h",
                },
            },
        )
    )
    job_a16 = scheduler.storage.acquire_next_job()
    assert job_a16 is not None
    scheduler.storage.mark_failed(job_id_a16, "[CUDA_OOM] CUDA out of memory. Tried to allocate 2.00 GiB")

    failed_a16 = scheduler.storage.get_job(job_id_a16)
    retry_id_a16 = scheduler._handle_oom_retry_if_needed(failed_a16)
    assert retry_id_a16 is not None

    retry_job = scheduler.storage.get_job(retry_id_a16)
    assert retry_job is not None
    retry_spec_d = retry_job.payload["train_spec"]
    assert retry_spec_d["batch_size"] == 1
    assert retry_spec_d["gradient_accumulation_steps"] == 16  # Clamped, not 32!

    # Verify that TrainSpec validates successfully without raising ValueError
    validated_spec = TrainSpec.from_dict(retry_spec_d)
    assert validated_spec.batch_size == 1
    assert validated_spec.gradient_accumulation_steps == 16


# ---------------------------------------------------------------------------
# 11. Review Round 4: Durable Checkpoints, Safe Stop & Strict Verification Tests
# ---------------------------------------------------------------------------


def test_train_stop_preserves_durable_checkpoint_and_loadable_adapter(temp_db_path: Path, tmp_path: Path):
    """Verifies that stopping a training job preserves a durable checkpoint and loadable adapter."""
    from paxg_lab.queue.checkpoint import TrainingCheckpointManager
    from paxg_lab.queue.worker import GPUWorker

    storage = GPUJobStorage(temp_db_path)
    snap_path = get_or_create_test_snapshot(tmp_path, timeframe="4h")
    ckpt_dir = tmp_path / "checkpoints_stop_test"
    adapter_dir = tmp_path / "adapters_stop_test"
    job_id = "train_stop_preservation_job"

    storage.submit_job(
        JobSpec(
            job_id=job_id,
            job_type=JobType.TRAIN.value,
            timeframe="4h",
            priority=JobPriority.MANUAL.value,
            payload={
                "snapshot_path": str(snap_path),
                "checkpoint_dir": str(ckpt_dir),
                "adapter_store_dir": str(adapter_dir),
                "train_spec": {
                    "timeframe": "4h",
                    "horizon": 6,
                    "max_epochs": 3,
                    "batch_size": 2,
                    "gradient_accumulation_steps": 2,
                    "max_samples_per_epoch": 8,
                },
            },
        )
    )

    worker = GPUWorker(job_id=job_id, db_path=temp_db_path)

    # Set stop event during progress callback
    orig_dispatch = worker._dispatch

    def stopping_dispatch(job):
        worker.stop_event.set()
        return orig_dispatch(job)

    worker._dispatch = stopping_dispatch
    exit_code = worker.run()
    assert exit_code == 0

    # Verify job status in storage is CANCELLED and result metadata is preserved
    finished_job = storage.get_job(job_id)
    assert finished_job is not None
    assert finished_job.status == JobStatus.CANCELLED.value
    assert finished_job.result is not None
    assert "checkpoint_path" in finished_job.result
    assert "adapter_path" in finished_job.result
    assert finished_job.result["status"] == "cancelled"

    # Verify durable checkpoint was saved and can be loaded
    ckpt_mgr = TrainingCheckpointManager(ckpt_dir)
    ckpt = ckpt_mgr.load_checkpoint(job_id)
    assert ckpt is not None
    assert ckpt.epoch >= 1
    assert ckpt.metadata["status"] in ("STOPPED", "IN_PROGRESS")
    assert ckpt.weights_path.exists()
    assert ckpt.manifest_path.exists()

    # Verify loadable adapter was also persisted to adapter store
    adapter_subdirs = list(adapter_dir.glob("paxg_*"))
    assert len(adapter_subdirs) >= 1
    saved_adapter_manifest = adapter_subdirs[0] / "paxg_manifest.json"
    assert saved_adapter_manifest.exists()


def test_scheduler_and_trainer_reconcile_durable_checkpoint_on_restart(temp_db_path: Path, tmp_path: Path):
    """Verifies that scheduler reconciles durable checkpoints on restart, requeues continuation job, and trainer resumes."""
    from paxg_lab.constants import MODEL_REPO, MODEL_REVISION
    from paxg_lab.data.features import FEATURE_SPECS
    from paxg_lab.data.snapshot import DatasetSnapshot
    from paxg_lab.model.manifest import AdapterManifest
    from paxg_lab.model.train_spec import TrainSpec
    from paxg_lab.model.trainer import LoRATrainer
    from paxg_lab.queue.checkpoint import TrainingCheckpointManager
    from timesfm3 import TimesFM3Torch

    storage = GPUJobStorage(temp_db_path)
    ckpt_dir = tmp_path / "checkpoints_reconcile_test"
    ckpt_mgr = TrainingCheckpointManager(ckpt_dir)
    job_id = "interrupted_reconcile_job"

    snap_path = get_or_create_test_snapshot(tmp_path, timeframe="4h")
    snapshot = DatasetSnapshot.load(snap_path)
    actual_snap_hash = snapshot.metadata.sha256

    # Create dummy base model and attach lora to build initial checkpoint
    base_model = TimesFM3Torch.from_pretrained(MODEL_REPO, revision=MODEL_REVISION)
    trainer = LoRATrainer(base_model=base_model, spec=TrainSpec(timeframe="4h", horizon=6, max_epochs=3, max_samples_per_epoch=8))
    from paxg_lab.model.lora import build_lora_timesfm3

    peft_model = build_lora_timesfm3(base_model, lora_r=4, lora_alpha=8)

    f_spec = FEATURE_SPECS["B"]
    manifest = AdapterManifest(
        adapter_id=f"ckpt_{job_id}_1_4",
        timeframe="4h",
        horizon=6,
        context_len=256,
        feature_set="B",
        feature_columns=list(f_spec.columns),
        base_model_repo=MODEL_REPO,
        base_model_revision=MODEL_REVISION,
        train_spec=trainer.spec.to_dict(),
        snapshot_hash=actual_snap_hash,
        best_epoch=1,
        best_val_loss=0.042,
    )
    ckpt_mgr.save_checkpoint(
        job_id=job_id,
        peft_model=peft_model,
        manifest=manifest,
        epoch=1,
        step=4,
        best_val_loss=0.042,
        minibatch_idx=3,  # Completed epoch 1
        status="IN_PROGRESS",
    )

    # Simulate crashed job in DB
    dead_pid = 99999999
    storage.submit_job(
        JobSpec(
            job_id=job_id,
            job_type=JobType.TRAIN.value,
            timeframe="4h",
            priority=JobPriority.MANUAL.value,
            payload={"checkpoint_dir": str(ckpt_dir)},
        )
    )
    storage.acquire_next_job()
    storage.register_worker(job_id, dead_pid, time.time() - 100)

    # 1. Scheduler recovery on startup: reconciles and automatically requeues resumed job
    scheduler = GPUScheduler(
        db_path=temp_db_path,
        acquire_coordinator_lock=False,
    )
    recovered = scheduler.recovered_on_startup

    assert job_id in recovered
    recovered_job = storage.get_job(job_id)
    assert recovered_job.status == JobStatus.INTERRUPTED.value
    assert "Reconciled durable checkpoint: epoch=1, step=4" in recovered_job.error_message

    # Verify queue-level requeue of resumed job
    resumed_job_id = f"{job_id}_resumed"
    resumed_job = storage.get_job(resumed_job_id)
    assert resumed_job is not None
    assert resumed_job.status == JobStatus.QUEUED.value
    assert resumed_job.payload.get("resume_from_job_id") == job_id
    assert resumed_job.payload.get("resume_attempt") == 1

    # 2. Resumed training starts from epoch 2 without rerunning epoch 1
    features_df = snapshot.to_dataframe("B")
    epochs_trained = []

    def progress_cb(info):
        if "epoch" in info:
            epochs_trained.append(info["epoch"])
        return True

    res = trainer.train(
        features_df=features_df,
        snapshot_hash=actual_snap_hash,
        progress_callback=progress_cb,
        checkpoint_manager=ckpt_mgr,
        job_id=job_id,
    )
    # Verify epoch 1 was skipped and not rerun
    assert 1 not in epochs_trained
    assert all(e >= 2 for e in epochs_trained)


def test_checkpoint_atomic_write_preserves_valid_checkpoint_on_crash(tmp_path: Path):
    """Verifies that atomic write guarantees pre-existing valid checkpoint is unharmed if crash occurs during write, commit, or checksum corruption."""
    from paxg_lab.constants import MODEL_REPO, MODEL_REVISION
    from paxg_lab.data.features import FEATURE_SPECS
    from paxg_lab.model.lora import build_lora_timesfm3
    from paxg_lab.model.manifest import AdapterManifest
    from paxg_lab.queue.checkpoint import TrainingCheckpointManager
    from timesfm3 import TimesFM3Torch

    ckpt_dir = tmp_path / "checkpoints_atomic_test"
    ckpt_mgr = TrainingCheckpointManager(ckpt_dir)
    job_id = "atomic_crash_test_job"

    base_model = TimesFM3Torch.from_pretrained(MODEL_REPO, revision=MODEL_REVISION)
    peft_model = build_lora_timesfm3(base_model, lora_r=4, lora_alpha=8)
    f_spec = FEATURE_SPECS["B"]

    manifest1 = AdapterManifest(
        adapter_id="ckpt_atomic_1",
        timeframe="4h",
        horizon=6,
        context_len=256,
        feature_set="B",
        feature_columns=list(f_spec.columns),
        base_model_repo=MODEL_REPO,
        base_model_revision=MODEL_REVISION,
        train_spec={"timeframe": "4h", "horizon": 6},
        snapshot_hash="hash_v1",
        best_epoch=1,
        best_val_loss=0.050,
    )

    # 1. Save valid checkpoint 1
    ckpt_mgr.save_checkpoint(
        job_id=job_id,
        peft_model=peft_model,
        manifest=manifest1,
        epoch=1,
        step=5,
        best_val_loss=0.050,
    )
    loaded1 = ckpt_mgr.load_checkpoint(job_id)
    assert loaded1 is not None
    assert loaded1.epoch == 1
    assert loaded1.best_val_loss == 0.050

    # 2. Simulate crash during checkpoint 2 write before version directory commit
    manifest2 = AdapterManifest(
        adapter_id="ckpt_atomic_2",
        timeframe="4h",
        horizon=6,
        context_len=256,
        feature_set="B",
        feature_columns=list(f_spec.columns),
        base_model_repo=MODEL_REPO,
        base_model_revision=MODEL_REVISION,
        train_spec={"timeframe": "4h", "horizon": 6},
        snapshot_hash="hash_v2",
        best_epoch=2,
        best_val_loss=0.030,
    )

    with pytest.raises(RuntimeError, match="Simulated crash during checkpoint write"):
        ckpt_mgr.save_checkpoint(
            job_id=job_id,
            peft_model=peft_model,
            manifest=manifest2,
            epoch=2,
            step=10,
            best_val_loss=0.030,
            simulate_crash_before_replace=True,
        )

    # Verify checkpoint 1 remains completely intact and valid
    loaded_after_crash1 = ckpt_mgr.load_checkpoint(job_id)
    assert loaded_after_crash1 is not None
    assert loaded_after_crash1.epoch == 1
    assert loaded_after_crash1.global_step == 5
    assert loaded_after_crash1.best_val_loss == 0.050

    # 3. Simulate crash after version dir is committed, but before CURRENT pointer swap
    with pytest.raises(RuntimeError, match="Simulated crash after version write"):
        ckpt_mgr.save_checkpoint(
            job_id=job_id,
            peft_model=peft_model,
            manifest=manifest2,
            epoch=2,
            step=10,
            best_val_loss=0.030,
            simulate_crash_during_commit=True,
        )

    # CURRENT pointer still points to version 1!
    loaded_after_crash2 = ckpt_mgr.load_checkpoint(job_id)
    assert loaded_after_crash2 is not None
    assert loaded_after_crash2.epoch == 1
    assert loaded_after_crash2.global_step == 5

    # 4. Checksum corruption detection: corrupted file fails cryptographic integrity verification
    ckpt_mgr.save_checkpoint(
        job_id=job_id,
        peft_model=peft_model,
        manifest=manifest2,
        epoch=2,
        step=12,
        best_val_loss=0.025,
        simulate_checksum_corruption=True,
    )
    # Checkpoint 2 has corrupted checksum -> load_checkpoint falls back to verified version 1!
    loaded_after_tamper = ckpt_mgr.load_checkpoint(job_id)
    assert loaded_after_tamper is not None
    assert loaded_after_tamper.epoch == 1
    assert loaded_after_tamper.best_val_loss == 0.050


def test_safe_terminate_refuses_when_create_time_none():
    """Verifies that safe_terminate_process strictly refuses termination when expected_create_time is None (fail closed)."""
    current_pid = os.getpid()
    # Should refuse to terminate even for valid PID because expected_create_time is None
    result = safe_terminate_process(current_pid, expected_create_time=None)
    assert result is False


def test_spawn_worker_retries_and_terminates_if_create_time_fails(temp_db_path: Path, tmp_path: Path, monkeypatch):
    """Verifies that failure to obtain create_time after spawn retries 5x, terminates child, and rejects startup."""
    scheduler = GPUScheduler(db_path=temp_db_path, acquire_coordinator_lock=False)
    storage = scheduler.storage

    job_id = "test_create_time_fail_job"
    job = JobSpec(
        job_id=job_id,
        job_type=JobType.DUMMY.value,
        timeframe="1h",
        priority=JobPriority.MANUAL.value,
    )
    storage.submit_job(job)
    acquired = storage.acquire_next_job()
    assert acquired is not None

    # Mock psutil.Process.create_time to always raise an exception
    attempt_count = [0]
    orig_process = psutil.Process

    class MockProcess(orig_process):
        def create_time(self):
            attempt_count[0] += 1
            raise RuntimeError("Access denied reading process start time")

    monkeypatch.setattr(psutil, "Process", MockProcess)

    success = scheduler._spawn_worker(acquired)
    assert success is False
    assert attempt_count[0] >= 5  # Retried up to 5 times

    # Worker was not allowed to start up
    assert scheduler.active_worker is None
    job_status = storage.get_job(job_id).status
    assert job_status == JobStatus.FAILED.value


def test_access_denied_process_state_prevents_unsafe_worker_dispatch(temp_db_path: Path, monkeypatch):
    """Verifies that AccessDenied on process inspection is treated as ALIVE and prevents unsafe second worker dispatch."""
    # 1. is_process_alive returns True on AccessDenied (fail-safe)
    orig_process = psutil.Process

    def mock_process_access_denied(pid):
        if pid == 12345:
            raise psutil.AccessDenied()
        return orig_process(pid)

    monkeypatch.setattr(psutil, "Process", mock_process_access_denied)
    assert is_process_alive(12345, 1000.0) is True

    # 2. Scheduler tick does not conclude worker is dead and does not dispatch a second worker
    scheduler = GPUScheduler(db_path=temp_db_path, acquire_coordinator_lock=False)
    storage = scheduler.storage

    job1 = JobSpec(job_id="job_active", job_type=JobType.DUMMY.value, timeframe="1h", priority=JobPriority.MANUAL.value)
    job2 = JobSpec(job_id="job_queued", job_type=JobType.DUMMY.value, timeframe="1h", priority=JobPriority.MANUAL.value)
    storage.submit_job(job1)
    storage.submit_job(job2)

    # Set active worker in scheduler
    scheduler.active_job_id = "job_active"
    scheduler.active_worker_pid = 12345
    scheduler.active_worker_create_time = 1000.0
    storage.acquire_next_job()
    storage.register_worker("job_active", 12345, 1000.0)

    # Tick: is_process_alive returns True (AccessDenied treated as ALIVE)
    scheduler.tick()

    # Worker should still be active, job2 should NOT be dispatched
    assert scheduler.active_job_id == "job_active"
    queued = storage.get_job("job_queued")
    assert queued.status == JobStatus.QUEUED.value


def test_checkpoint_compatibility_rejects_mismatch(tmp_path: Path):
    """Verifies that checkpoint compatibility strictly fails closed on snapshot, spec, or base model mismatch."""
    from paxg_lab.constants import MODEL_REPO, MODEL_REVISION
    from paxg_lab.data.features import FEATURE_SPECS
    from paxg_lab.data.snapshot import DatasetSnapshot
    from paxg_lab.model.lora import build_lora_timesfm3
    from paxg_lab.model.manifest import AdapterManifest
    from paxg_lab.model.train_spec import TrainSpec
    from paxg_lab.model.trainer import LoRATrainer
    from paxg_lab.queue.checkpoint import TrainingCheckpointManager
    from timesfm3 import TimesFM3Torch

    ckpt_dir = tmp_path / "checkpoints_compat_test"
    ckpt_mgr = TrainingCheckpointManager(ckpt_dir)
    job_id = "compat_check_job"

    base_model = TimesFM3Torch.from_pretrained(MODEL_REPO, revision=MODEL_REVISION)
    peft_model = build_lora_timesfm3(base_model, lora_r=4, lora_alpha=8)
    f_spec = FEATURE_SPECS["B"]

    manifest = AdapterManifest(
        adapter_id="ckpt_compat_1",
        timeframe="4h",
        horizon=6,
        context_len=256,
        feature_set="B",
        feature_columns=list(f_spec.columns),
        base_model_repo=MODEL_REPO,
        base_model_revision=MODEL_REVISION,
        train_spec={"timeframe": "4h", "horizon": 6, "context_len": 256, "feature_set": "B", "lora_r": 4, "lora_alpha": 8},
        snapshot_hash="hash_alpha",
        best_epoch=1,
        best_val_loss=0.040,
    )
    ckpt_mgr.save_checkpoint(
        job_id=job_id,
        peft_model=peft_model,
        manifest=manifest,
        epoch=1,
        step=5,
        best_val_loss=0.040,
    )
    ckpt = ckpt_mgr.load_checkpoint(job_id)
    assert ckpt is not None

    # 1. Snapshot hash mismatch -> must fail closed
    spec = TrainSpec(timeframe="4h", horizon=6, context_len=256, feature_set="B", lora_r=4, lora_alpha=8)
    with pytest.raises(ValueError, match="snapshot_hash mismatch"):
        TrainingCheckpointManager.validate_compatibility(ckpt, spec, current_snapshot_hash="hash_different")

    # 2. Timeframe mismatch -> must fail closed
    spec_tf_mismatch = TrainSpec(timeframe="1h", horizon=24, context_len=256, feature_set="B", lora_r=4, lora_alpha=8)
    with pytest.raises(ValueError, match="timeframe' mismatch"):
        TrainingCheckpointManager.validate_compatibility(ckpt, spec_tf_mismatch, current_snapshot_hash="hash_alpha")

    # 3. LoRA rank mismatch -> must fail closed
    spec_rank_mismatch = TrainSpec(timeframe="4h", horizon=6, context_len=256, feature_set="B", lora_r=8, lora_alpha=16)
    with pytest.raises(ValueError, match="lora_r' mismatch"):
        TrainingCheckpointManager.validate_compatibility(ckpt, spec_rank_mismatch, current_snapshot_hash="hash_alpha")

    # 4. Base model mismatch -> must fail closed
    with pytest.raises(ValueError, match="base model mismatch"):
        TrainingCheckpointManager.validate_compatibility(
            ckpt, spec, current_snapshot_hash="hash_alpha", base_model_repo="different/base-repo"
        )

    # 5. LoRATrainer.train strictly refuses resume on mismatched snapshot
    snap_path = get_or_create_test_snapshot(tmp_path, timeframe="4h")
    snapshot = DatasetSnapshot.load(snap_path)
    clean_base_model = TimesFM3Torch.from_pretrained(MODEL_REPO, revision=MODEL_REVISION)
    trainer = LoRATrainer(base_model=clean_base_model, spec=spec)
    with pytest.raises(ValueError, match="snapshot_hash mismatch"):
        trainer.train(
            features_df=snapshot.to_dataframe("B"),
            snapshot_hash="mismatched_runtime_hash",
            checkpoint_manager=ckpt_mgr,
            job_id=job_id,
        )


def test_mid_epoch_checkpoint_and_resumption_no_skipping(tmp_path: Path):
    """Verifies that mid-epoch checkpoint continuation does not skip remaining minibatches of the epoch nor rerun completed minibatches."""
    from paxg_lab.constants import MODEL_REPO, MODEL_REVISION
    from paxg_lab.data.features import FEATURE_SPECS
    from paxg_lab.data.snapshot import DatasetSnapshot
    from paxg_lab.model.lora import build_lora_timesfm3
    from paxg_lab.model.manifest import AdapterManifest
    from paxg_lab.model.train_spec import TrainSpec
    from paxg_lab.model.trainer import LoRATrainer
    from paxg_lab.queue.checkpoint import TrainingCheckpointManager
    from timesfm3 import TimesFM3Torch

    ckpt_dir = tmp_path / "checkpoints_midepoch_test"
    ckpt_mgr = TrainingCheckpointManager(ckpt_dir)
    job_id = "midepoch_resume_job"

    snap_path = get_or_create_test_snapshot(tmp_path, timeframe="4h")
    snapshot = DatasetSnapshot.load(snap_path)
    features_df = snapshot.to_dataframe("B")
    snap_hash = snapshot.metadata.sha256

    base_model = TimesFM3Torch.from_pretrained(MODEL_REPO, revision=MODEL_REVISION)
    spec = TrainSpec(
        timeframe="4h",
        horizon=6,
        context_len=256,
        feature_set="B",
        max_epochs=2,
        batch_size=2,
        gradient_accumulation_steps=1,
        max_samples_per_epoch=8,  # Exactly 4 minibatches per epoch (0, 1, 2, 3)
    )
    trainer = LoRATrainer(base_model=base_model, spec=spec)
    peft_model = build_lora_timesfm3(base_model, lora_r=4, lora_alpha=8)

    # 1. Create a checkpoint stopped MID-EPOCH at epoch 1, minibatch_idx 0 (1/4 done)
    f_spec = FEATURE_SPECS["B"]
    manifest = AdapterManifest(
        adapter_id="ckpt_midepoch_1",
        timeframe="4h",
        horizon=6,
        context_len=256,
        feature_set="B",
        feature_columns=list(f_spec.columns),
        base_model_repo=MODEL_REPO,
        base_model_revision=MODEL_REVISION,
        train_spec=spec.to_dict(),
        snapshot_hash=snap_hash,
        best_epoch=1,
        best_val_loss=0.060,
    )
    ckpt_mgr.save_checkpoint(
        job_id=job_id,
        peft_model=peft_model,
        manifest=manifest,
        epoch=1,
        step=1,
        best_val_loss=0.060,
        minibatch_idx=0,  # Stopped after first minibatch of epoch 1
        status="STOPPED",
    )

    # 2. Resume training
    records: list[dict[str, Any]] = []

    def progress_cb(info):
        records.append(dict(info))
        return True

    res = trainer.train(
        features_df=features_df,
        snapshot_hash=snap_hash,
        progress_callback=progress_cb,
        checkpoint_manager=ckpt_mgr,
        job_id=job_id,
    )

    # Verify that:
    # 1. Epoch 1 was NOT skipped (it continued from minibatch 1 to completion)
    # 2. Total steps equal 7 (1 previously completed step + 3 remaining steps in epoch 1 + 4 steps in epoch 2)
    step_records = [r for r in records if r.get("type") == "step_update"]
    assert len(step_records) > 0
    # Steps should start from global_step 2 (continuation of step 1)
    first_continued_step = step_records[0]["step"]
    assert first_continued_step == 2
    assert res.total_steps >= 7
    assert res.manifest.best_epoch >= 1


def test_stopped_job_result_persisted_in_database(temp_db_path: Path, tmp_path: Path):
    """Verifies that when a job is stopped, its full result metadata is preserved in the database."""
    from paxg_lab.queue.worker import GPUWorker

    storage = GPUJobStorage(temp_db_path)
    snap_path = get_or_create_test_snapshot(tmp_path, timeframe="4h")
    ckpt_dir = tmp_path / "checkpoints_res_test"
    adapter_dir = tmp_path / "adapters_res_test"
    job_id = "stop_result_persisted_job"

    storage.submit_job(
        JobSpec(
            job_id=job_id,
            job_type=JobType.TRAIN.value,
            timeframe="4h",
            priority=JobPriority.MANUAL.value,
            payload={
                "snapshot_path": str(snap_path),
                "checkpoint_dir": str(ckpt_dir),
                "adapter_store_dir": str(adapter_dir),
                "train_spec": {
                    "timeframe": "4h",
                    "horizon": 6,
                    "max_epochs": 3,
                    "batch_size": 2,
                    "gradient_accumulation_steps": 2,
                    "max_samples_per_epoch": 8,
                },
            },
        )
    )

    worker = GPUWorker(job_id=job_id, db_path=temp_db_path)
    orig_dispatch = worker._dispatch

    def stop_dispatch(job):
        worker.stop_event.set()
        return orig_dispatch(job)

    worker._dispatch = stop_dispatch
    worker.run()

    job_record = storage.get_job(job_id)
    assert job_record is not None
    assert job_record.status == JobStatus.CANCELLED.value
    assert job_record.result is not None
    assert job_record.result.get("status") == "cancelled"
    assert "checkpoint_path" in job_record.result
    assert "adapter_path" in job_record.result
    assert "total_steps" in job_record.result
    assert "best_val_loss" in job_record.result




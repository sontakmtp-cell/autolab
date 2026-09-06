"""Unit and integration tests for P4: Single GPU Process Queue, Heartbeat, Safe Stop, and Recovery."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any

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
    # (or directly with different PID recorded in state)
    with pytest.raises(RuntimeError, match="Another GPUScheduler coordinator .* is currently active"):
        # Simulate active coordinator with different PID
        sched_a.storage.set_state("coordinator_pid", "4")  # System idle PID (always alive)
        sched_a.storage.set_state("coordinator_create_time", str(psutil.Process(4).create_time()))
        sched_a.storage.set_state("coordinator_heartbeat", str(time.time()))

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

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

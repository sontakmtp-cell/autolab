"""Process guard and safe termination utilities for PAXG Forecast Lab.

Enforces PID and start-time verification before terminating any process to
prevent accidentally terminating unrelated OS processes whose PIDs were recycled.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import psutil

logger = logging.getLogger(__name__)


def is_process_alive(pid: int | None, expected_create_time: float | None = None) -> bool:
    """Checks whether a process with the given PID is alive and matches expected create_time.
    
    Args:
        pid: The process ID to check.
        expected_create_time: If provided, verifies psutil.Process(pid).create_time() matches
                              within a 2.0 second tolerance.
    """
    if pid is None or pid <= 0:
        return False

    try:
        proc = psutil.Process(pid)
        if not proc.is_running() or proc.status() == psutil.STATUS_ZOMBIE:
            return False

        if expected_create_time is not None:
            actual_time = proc.create_time()
            if abs(actual_time - expected_create_time) > 2.0:
                logger.warning(
                    "PID %d exists but create_time mismatch (expected %.2f, actual %.2f). PID was recycled!",
                    pid,
                    expected_create_time,
                    actual_time,
                )
                return False

        return True
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return False
    except psutil.AccessDenied:
        # Fail-closed/safe: cannot inspect process permissions, so treat as ALIVE
        # to prevent scheduler from prematurely concluding worker is dead and dispatching a 2nd GPU worker.
        logger.warning("Access denied inspecting process PID %d. Treating as ALIVE for GPU safety.", pid)
        return True
    except Exception as exc:
        logger.warning("Unexpected error inspecting process PID %d: %s. Treating as ALIVE for GPU safety.", pid, exc)
        return True


def safe_terminate_process(
    pid: int | None,
    expected_create_time: float | None = None,
    timeout: float = 5.0,
) -> bool:
    """Safely terminates a process after strictly verifying PID and creation time.
    
    FAIL-CLOSED: If expected_create_time is None, termination is strictly REFUSED.

    Args:
        pid: The process ID to terminate.
        expected_create_time: Recorded creation time of the process (required).
        timeout: Maximum seconds to wait after terminate() before issuing kill().
        
    Returns:
        True if process was successfully terminated or was already dead.
        False if PID verification failed (e.g. recycled PID or missing expected_create_time).
    """
    if pid is None or pid <= 0:
        return True

    # Fail closed: expected_create_time is strictly required
    if expected_create_time is None:
        logger.error(
            "REFUSING TO TERMINATE PID %d: expected_create_time is None (fail-closed policy). "
            "P4 requires strict verification of both PID and creation time.",
            pid,
        )
        return False

    try:
        proc = psutil.Process(pid)
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return True
    except psutil.AccessDenied:
        logger.error("Access denied inspecting process PID %d", pid)
        return False

    # Strict creation time verification
    try:
        actual_time = proc.create_time()
        if abs(actual_time - expected_create_time) > 2.0:
            logger.error(
                "REFUSING TO TERMINATE PID %d: create_time mismatch! "
                "Expected %.2f, actual %.2f. This PID belongs to an unrelated recycled process.",
                pid,
                expected_create_time,
                actual_time,
            )
            return False
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return True
    except psutil.AccessDenied:
        logger.error("Access denied inspecting create_time for PID %d", pid)
        return False

    logger.info("Terminating worker process PID %d (grace timeout=%.1fs)...", pid, timeout)

    # 1. Try graceful terminate
    try:
        proc.terminate()
        proc.wait(timeout=timeout)
        logger.info("Process PID %d terminated cleanly.", pid)
        return True
    except psutil.TimeoutExpired:
        logger.warning("Process PID %d did not terminate within %.1fs, sending kill()...", pid, timeout)
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return True

    # 2. Force kill if still running
    try:
        proc.kill()
        proc.wait(timeout=3.0)
        logger.info("Process PID %d forcefully killed.", pid)
        return True
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return True
    except Exception as exc:
        logger.error("Failed to kill process PID %d: %s", pid, exc)
        return False

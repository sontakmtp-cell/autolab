"""Job specifications, status states, and priority conventions for GPU Queue in PAXG Forecast Lab."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import json
import time
from typing import Any


class JobStatus(str, Enum):
    """Lifecycle status of a GPU job.
    
    Allowed transitions:
    QUEUED -> RUNNING -> SUCCEEDED / FAILED / CANCELLED / INTERRUPTED
    QUEUED -> CANCELLED
    """

    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    INTERRUPTED = "INTERRUPTED"

    def is_terminal(self) -> bool:
        """Returns True if this status is a completed terminal state."""
        return self in (
            JobStatus.SUCCEEDED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
            JobStatus.INTERRUPTED,
        )


class JobPriority(int, Enum):
    """Priority hierarchy strictly mandated by PLAN.md section 4.1:
    
    1: Instant Forecast (Highest)
    2: Manual Work (Train / Backtest)
    3: Auto Tuning / Optimization (Lowest)
    """

    FORECAST = 1
    MANUAL = 2
    AUTO = 3


class JobType(str, Enum):
    """Types of jobs processed by the single GPU worker."""

    FORECAST = "FORECAST"
    TRAIN = "TRAIN"
    BACKTEST = "BACKTEST"
    AUTO_TRIAL = "AUTO_TRIAL"
    DUMMY = "DUMMY"  # For testing and controlled failure verification


class AutoRunState(str, Enum):
    """State machine for autonomous optimization per timeframe (PLAN 4.2)."""

    SEARCHING = "SEARCHING"
    VALIDATING = "VALIDATING"
    WAITING_DATA = "WAITING_DATA"
    WAITING_AUDIT = "WAITING_AUDIT"
    PAUSED_ERROR = "PAUSED_ERROR"
    STOPPED = "STOPPED"


@dataclass
class JobSpec:
    """Specification, execution metadata, and telemetry for a single GPU task."""

    job_id: str
    job_type: str
    timeframe: str | None = None  # "1h", "4h", or None
    priority: int = JobPriority.AUTO.value
    status: str = JobStatus.QUEUED.value
    payload: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] | None = None
    error_message: str | None = None
    idempotency_key: str | None = None
    worker_pid: int | None = None
    worker_create_time: float | None = None
    timeout_seconds: float = 1200.0  # 20 minutes limit for auto jobs
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    heartbeat_at: float | None = None
    cancel_requested: bool = False
    progress_pct: float = 0.0
    progress_message: str = ""

    def __post_init__(self) -> None:
        # Validate status enum
        valid_statuses = {s.value for s in JobStatus}
        if self.status not in valid_statuses:
            raise ValueError(f"Invalid JobStatus '{self.status}'. Must be one of {valid_statuses}.")

        # Validate priority
        valid_priorities = {p.value for p in JobPriority}
        if self.priority not in valid_priorities:
            raise ValueError(f"Invalid JobPriority '{self.priority}'. Must be one of {valid_priorities}.")

        # Validate timeframe if specified
        if self.timeframe is not None and self.timeframe not in ("1h", "4h"):
            raise ValueError(f"Invalid timeframe '{self.timeframe}'. Must be '1h', '4h', or None.")

        # Ensure timeout_seconds is positive
        if self.timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be positive, got {self.timeout_seconds}.")

    def to_dict(self) -> dict[str, Any]:
        """Serializes JobSpec to dictionary."""
        return asdict(self)

    @classmethod
    def from_row(cls, row: dict[str, Any] | tuple[Any, ...]) -> JobSpec:
        """Constructs JobSpec from an SQLite query row dictionary or tuple."""
        if isinstance(row, dict):
            payload = row["payload"]
            if isinstance(payload, str):
                payload = json.loads(payload)
            result = row.get("result")
            if isinstance(result, str) and result:
                result = json.loads(result)
            elif not result:
                result = None

            return cls(
                job_id=row["job_id"],
                job_type=row["job_type"],
                timeframe=row.get("timeframe"),
                priority=int(row["priority"]),
                status=row["status"],
                payload=payload or {},
                result=result,
                error_message=row.get("error_message"),
                idempotency_key=row.get("idempotency_key"),
                worker_pid=row.get("worker_pid"),
                worker_create_time=row.get("worker_create_time"),
                timeout_seconds=float(row.get("timeout_seconds", 1200.0)),
                created_at=float(row["created_at"]),
                started_at=float(row["started_at"]) if row.get("started_at") is not None else None,
                finished_at=float(row["finished_at"]) if row.get("finished_at") is not None else None,
                heartbeat_at=float(row["heartbeat_at"]) if row.get("heartbeat_at") is not None else None,
                cancel_requested=bool(row.get("cancel_requested", 0)),
                progress_pct=float(row.get("progress_pct", 0.0)),
                progress_message=row.get("progress_message") or "",
            )

        raise TypeError(f"Unsupported row type: {type(row)}")

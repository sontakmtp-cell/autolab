"""Adapter manifest data structure, serialization, and compatibility verification."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from ..constants import (
    MODEL_REPO,
    MODEL_REVISION,
    get_horizon_for_timeframe,
)


@dataclass
class AdapterManifest:
    """Immutable manifest accompanying every trained LoRA adapter.

    Contains complete provenance, hyperparameter specifications, base model revision,
    training time range, snapshot hashes, and SHA-256 integrity checksums.
    """

    adapter_id: str
    timeframe: str
    horizon: int
    context_len: int
    feature_set: str
    feature_columns: list[str]
    base_model_repo: str = MODEL_REPO
    base_model_revision: str = MODEL_REVISION
    lora_config: dict[str, Any] = field(default_factory=dict)
    train_spec: dict[str, Any] = field(default_factory=dict)
    training_range: dict[str, Any] = field(default_factory=dict)
    snapshot_hash: str = ""
    best_epoch: int = 0
    best_val_loss: float = 0.0
    metrics: dict[str, Any] = field(default_factory=dict)
    file_hashes: dict[str, str] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    code_version: str = "1.0.0"
    is_verified: bool = False

    def __post_init__(self) -> None:
        """Enforces horizon convention and normalizes fields."""
        tf = self.timeframe.lower().strip()
        expected_horizon = get_horizon_for_timeframe(tf)
        if self.horizon != expected_horizon:
            raise ValueError(
                f"AdapterManifest horizon mismatch: timeframe '{tf}' expects horizon={expected_horizon}, "
                f"but got {self.horizon}."
            )
        self.timeframe = tf
        self.feature_set = self.feature_set.upper().strip()

    def to_dict(self) -> dict[str, Any]:
        """Converts manifest to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AdapterManifest:
        """Constructs AdapterManifest from dictionary."""
        clean_data = dict(data)
        # Backward compatibility if extra fields exist
        valid_fields = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in clean_data.items() if k in valid_fields}
        return cls(**filtered)

    def save_json(self, file_path: str | Path) -> Path:
        """Saves manifest as formatted JSON."""
        p = Path(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        return p

    @classmethod
    def load_json(cls, file_path: str | Path) -> AdapterManifest:
        """Loads AdapterManifest from JSON file."""
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"Adapter manifest not found: {p}")
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def verify_compatibility(
        self,
        expected_timeframe: str | None = None,
        expected_horizon: int | None = None,
        expected_context: int | None = None,
        expected_feature_set: str | None = None,
        expected_columns: list[str] | None = None,
        expected_num_features: int | None = None,
        expected_base_revision: str | None = None,
    ) -> None:
        """Strictly validates compatibility against expected deployment parameters.

        Raises ValueError with clear diagnostics if incompatible.
        """
        # Timeframe
        if expected_timeframe is not None:
            etf = expected_timeframe.lower().strip()
            if self.timeframe != etf:
                raise ValueError(
                    f"Incompatible timeframe: adapter trained for '{self.timeframe}', "
                    f"requested '{etf}'."
                )

        # Horizon
        if expected_horizon is not None:
            if self.horizon != expected_horizon:
                raise ValueError(
                    f"Incompatible horizon: adapter trained for horizon={self.horizon}, "
                    f"requested {expected_horizon}."
                )

        # Context length
        if expected_context is not None:
            if self.context_len != expected_context:
                raise ValueError(
                    f"Incompatible context length: adapter trained for context_len={self.context_len}, "
                    f"requested {expected_context}."
                )

        # Feature set
        if expected_feature_set is not None:
            efs = expected_feature_set.upper().strip()
            if self.feature_set != efs:
                raise ValueError(
                    f"Incompatible feature set: adapter trained for set '{self.feature_set}', "
                    f"requested '{efs}'."
                )

        # Feature columns
        if expected_columns is not None:
            if list(self.feature_columns) != list(expected_columns):
                raise ValueError(
                    f"Incompatible feature columns:\n"
                    f"  Adapter columns: {self.feature_columns}\n"
                    f"  Requested columns: {expected_columns}"
                )

        # Number of features
        if expected_num_features is not None:
            if len(self.feature_columns) != expected_num_features:
                raise ValueError(
                    f"Incompatible number of features: adapter expects {len(self.feature_columns)}, "
                    f"got {expected_num_features}."
                )

        # Base model revision
        if expected_base_revision is not None:
            if self.base_model_revision != expected_base_revision:
                raise ValueError(
                    f"Base model revision mismatch: adapter built for revision '{self.base_model_revision}', "
                    f"active model revision is '{expected_base_revision}'."
                )

    def check_in_sample_overlap(
        self,
        eval_start_ms: int,
        eval_end_ms: int,
    ) -> tuple[bool, str]:
        """Checks whether an evaluation window overlaps the adapter's training time range.

        Args:
            eval_start_ms: Start timestamp in ms of evaluation period.
            eval_end_ms: End timestamp in ms of evaluation period.

        Returns:
            Tuple of (is_overlapping: bool, warning_message: str).
        """
        train_start = self.training_range.get("start_time_ms")
        train_end = self.training_range.get("end_time_ms")
        val_start = self.training_range.get("val_start_time_ms")
        val_end = self.training_range.get("val_end_time_ms")

        starts = [int(s) for s in (train_start, val_start) if s is not None]
        ends = [int(e) for e in (train_end, val_end) if e is not None]

        if not starts or not ends:
            return (False, "Manifest training_range does not contain timestamp boundaries.")

        in_sample_start = min(starts)
        in_sample_end = max(ends)

        # Overlap condition: max(eval_start_ms, in_sample_start) <= min(eval_end_ms, in_sample_end)
        if max(eval_start_ms, in_sample_start) <= min(eval_end_ms, in_sample_end):
            overlap_start = max(eval_start_ms, in_sample_start)
            overlap_end = min(eval_end_ms, in_sample_end)
            overlap_hours = (overlap_end - overlap_start) / (1000 * 3600)
            msg = (
                f"IN-SAMPLE OVERLAP DETECTED: Evaluation period [{eval_start_ms}, {eval_end_ms}] "
                f"overlaps adapter training/validation range [{in_sample_start}, {in_sample_end}] "
                f"by {overlap_hours:.1f} hours ({overlap_start} to {overlap_end}). "
                "Evaluation on in-sample / parameter-selection data yields biased performance scores!"
            )
            return (True, msg)

        return (False, "Evaluation period is strictly out-of-sample relative to adapter training and validation data.")

"""Training specification and hyperparameter validation for TimesFM 3.0 LoRA."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from ..constants import (
    ALLOWED_CONTEXT_LENGTHS,
    DEFAULT_CONTEXT_LENGTH,
    get_horizon_for_timeframe,
)


@dataclass
class TrainSpec:
    """Hyperparameters and configuration for manual LoRA fine-tuning.

    All ranges and defaults strictly conform to PAXG Forecast Lab PLAN.md Section 3.2.
    """

    timeframe: Literal["1h", "4h"] = "1h"
    horizon: int | None = None
    feature_set: Literal["A", "B", "C"] = "B"
    context_len: int = DEFAULT_CONTEXT_LENGTH
    lora_r: int = 4
    lora_alpha: int | None = None
    lora_dropout: float = 0.10
    learning_rate: float = 5e-5
    max_epochs: int = 5
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    weight_decay: float = 0.01
    early_stopping_patience: int = 2
    grad_clip_norm: float = 1.0
    history_days: int | str = 365
    seed: int = 42
    extended_targets: bool = False
    custom_targets: list[str] | None = None
    max_samples_per_epoch: int = 1024
    warmup_ratio: float = 0.10

    def __post_init__(self) -> None:
        """Enforces strict hyperparameter constraints and horizon pairs per PLAN.md Section 3.2."""
        # 1. Timeframe & Horizon pair locking
        tf = str(self.timeframe).lower().strip()
        if tf not in ("1h", "4h"):
            raise ValueError(f"Unsupported timeframe '{self.timeframe}'. Must be '1h' or '4h'.")
        self.timeframe = tf  # type: ignore

        expected_horizon = get_horizon_for_timeframe(self.timeframe)
        if self.horizon is None:
            self.horizon = expected_horizon
        elif self.horizon != expected_horizon:
            raise ValueError(
                f"Horizon mismatch for timeframe '{self.timeframe}': "
                f"expected {expected_horizon}, got {self.horizon}. "
                "Per mandatory convention: 1h must use horizon=24, 4h must use horizon=6."
            )

        # 2. Feature Set
        fs = str(self.feature_set).upper().strip()
        if fs not in ("A", "B", "C"):
            raise ValueError(f"Unsupported feature set '{self.feature_set}'. Must be 'A', 'B', or 'C'.")
        self.feature_set = fs  # type: ignore

        # 3. Context Length (PLAN 3.2: 128, 256, 512)
        if self.context_len not in ALLOWED_CONTEXT_LENGTHS:
            raise ValueError(
                f"Context length {self.context_len} not allowed. "
                f"Must be one of {ALLOWED_CONTEXT_LENGTHS}."
            )

        # 4. LoRA Rank & Alpha (PLAN 3.2: 2, 4, 8, 16; default alpha = 2 * r)
        if self.lora_r not in (2, 4, 8, 16):
            raise ValueError(f"LoRA rank {self.lora_r} invalid. Must be one of (2, 4, 8, 16).")
        if self.lora_alpha is None:
            self.lora_alpha = 2 * self.lora_r
        elif self.lora_alpha <= 0:
            raise ValueError(f"LoRA alpha must be positive, got {self.lora_alpha}.")

        # 5. Dropout (PLAN 3.2: 0–0.20)
        if not (0.0 <= self.lora_dropout <= 0.20):
            raise ValueError(f"LoRA dropout {self.lora_dropout} out of bounds [0.0, 0.20].")

        # 6. Learning Rate (PLAN 3.2: 1e-5 to 3e-4)
        if not (1e-5 <= self.learning_rate <= 3e-4):
            raise ValueError(
                f"Learning rate {self.learning_rate} out of PLAN 3.2 bounds [1e-5, 3e-4]."
            )

        # 7. Epochs & Batching (PLAN 3.2: max_epochs 1–10, batch_size 1, 2, 4, grad_accum 1–16)
        if not (1 <= self.max_epochs <= 10):
            raise ValueError(f"max_epochs must be between 1 and 10 per PLAN 3.2, got {self.max_epochs}.")
        if self.batch_size not in (1, 2, 4):
            raise ValueError(f"Batch size {self.batch_size} must be one of (1, 2, 4) per PLAN 3.2.")
        if not (1 <= self.gradient_accumulation_steps <= 16):
            raise ValueError(
                f"gradient_accumulation_steps must be between 1 and 16 per PLAN 3.2, "
                f"got {self.gradient_accumulation_steps}."
            )

        # 8. Weight Decay & Early Stopping (PLAN 3.2: weight_decay 0–0.10, patience 1–4)
        if not (0.0 <= self.weight_decay <= 0.10):
            raise ValueError(f"weight_decay {self.weight_decay} out of bounds [0.0, 0.10].")
        if not (1 <= self.early_stopping_patience <= 4):
            raise ValueError(
                f"early_stopping_patience must be between 1 and 4 per PLAN 3.2, "
                f"got {self.early_stopping_patience}."
            )

        # 9. Gradient Clipping (PLAN 3.2: 0.5–2.0)
        if not (0.5 <= self.grad_clip_norm <= 2.0):
            raise ValueError(f"grad_clip_norm {self.grad_clip_norm} out of bounds [0.5, 2.0].")

        # 10. History Days (PLAN 3.2: 180 ngày, 365 ngày, toàn bộ)
        if isinstance(self.history_days, str):
            h_str = self.history_days.strip().lower()
            if h_str in ("all", "toan_bo", "toàn bộ"):
                self.history_days = "all"
            elif h_str in ("180", "365"):
                self.history_days = int(h_str)
            else:
                raise ValueError(
                    f"history_days '{self.history_days}' not allowed by PLAN 3.2. Must be 180, 365, or 'all'."
                )
        elif isinstance(self.history_days, int):
            if self.history_days not in (180, 365):
                raise ValueError(
                    f"history_days {self.history_days} not allowed by PLAN 3.2. Must be 180, 365, or 'all'."
                )
        else:
            raise ValueError(f"Invalid history_days type: {type(self.history_days)}")

        # 11. Warmup & Sampling
        if not (0.0 <= self.warmup_ratio <= 0.5):
            raise ValueError(f"warmup_ratio {self.warmup_ratio} out of bounds [0.0, 0.5].")
        if self.max_samples_per_epoch <= 0:
            raise ValueError(f"max_samples_per_epoch must be > 0, got {self.max_samples_per_epoch}.")

    @property
    def effective_batch_size(self) -> int:
        """Effective batch size = GPU batch size * gradient accumulation steps."""
        return self.batch_size * self.gradient_accumulation_steps

    def to_dict(self) -> dict[str, Any]:
        """Converts TrainSpec to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainSpec:
        """Instantiates TrainSpec from dictionary."""
        return cls(**data)

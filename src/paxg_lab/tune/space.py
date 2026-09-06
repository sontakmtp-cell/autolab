"""Search space definitions, invariant validation, and trial sampling for Optuna TPE.

Conforms strictly to PAXG Forecast Lab PLAN.md Section 3.6:
- context_len: [128, 256, 512]
- lora_r: [4, 8, 16]
- lora_alpha: strictly 2 * rank
- learning_rate: [1e-5, 2e-4] (log scale)
- lora_dropout: [0.0, 0.20] (step 0.05)
- weight_decay: [0.0, 0.05] (step 0.01)
- history_days: [180, 365, "all"]
- feature_set: ["B", "C"]
- target_modules_group: ["setB", "all"]
- horizon: strictly fixed by timeframe (1h -> 24, 4h -> 6)
- batch_size: 2, gradient_accumulation_steps: 8 (effective batch = 16)
- max_epochs: 10, early_stopping_patience: 3
"""

from __future__ import annotations

from typing import Any
import optuna

from ..constants import get_horizon_for_timeframe
from ..model.train_spec import TrainSpec


ALLOWED_CONTEXTS = [128, 256, 512]
ALLOWED_RANKS = [4, 8, 16]
ALLOWED_FEATURE_SETS = ["B", "C"]
ALLOWED_HISTORY_DAYS = [180, 365, "all"]
ALLOWED_TARGET_MODULE_GROUPS = ["setB", "all"]

LR_MIN = 1e-5
LR_MAX = 2e-4
DROPOUT_MIN = 0.0
DROPOUT_MAX = 0.20
WEIGHT_DECAY_MIN = 0.0
WEIGHT_DECAY_MAX = 0.05
MAX_EPOCHS_AUTO = 10
EARLY_STOPPING_PATIENCE_AUTO = 3
BATCH_SIZE_AUTO = 2
GRAD_ACCUM_AUTO = 8  # Effective batch = 16


def suggest_trial_spec(trial: optuna.Trial, timeframe: str, seed: int = 42) -> TrainSpec:
    """Samples hyperparameters from the strictly bounded search space for a single Optuna trial."""
    tf = str(timeframe).lower().strip()
    if tf not in ("1h", "4h"):
        raise ValueError(f"Unsupported timeframe '{timeframe}'. Must be '1h' or '4h'.")

    # Fixed horizon per timeframe - strictly NOT sampled per PLAN 3.6
    horizon = get_horizon_for_timeframe(tf)

    context_len = trial.suggest_categorical("context_len", ALLOWED_CONTEXTS)
    lora_r = trial.suggest_categorical("lora_r", ALLOWED_RANKS)
    # Alpha is strictly 2 * rank per PLAN 3.6
    lora_alpha = 2 * lora_r

    learning_rate = trial.suggest_float("learning_rate", LR_MIN, LR_MAX, log=True)
    lora_dropout = trial.suggest_float("lora_dropout", DROPOUT_MIN, DROPOUT_MAX, step=0.05)
    weight_decay = trial.suggest_float("weight_decay", WEIGHT_DECAY_MIN, WEIGHT_DECAY_MAX, step=0.01)

    feature_set = trial.suggest_categorical("feature_set", ALLOWED_FEATURE_SETS)
    history_days = trial.suggest_categorical("history_days", ALLOWED_HISTORY_DAYS)

    target_group = trial.suggest_categorical("target_modules_group", ALLOWED_TARGET_MODULE_GROUPS)
    extended_targets = (target_group == "all")

    spec = TrainSpec(
        timeframe=tf,  # type: ignore
        horizon=horizon,
        feature_set=feature_set,  # type: ignore
        context_len=context_len,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        learning_rate=learning_rate,
        max_epochs=MAX_EPOCHS_AUTO,
        batch_size=BATCH_SIZE_AUTO,
        gradient_accumulation_steps=GRAD_ACCUM_AUTO,
        weight_decay=weight_decay,
        early_stopping_patience=EARLY_STOPPING_PATIENCE_AUTO,
        history_days=history_days,
        seed=seed,
        extended_targets=extended_targets,
    )

    validate_train_spec_invariants(spec)
    return spec


def validate_train_spec_invariants(spec: TrainSpec) -> None:
    """Validates that a TrainSpec satisfies all mandatory PLAN 3.6 optimization invariants."""
    # 1. Horizon strictly matches timeframe
    expected_horizon = get_horizon_for_timeframe(spec.timeframe)
    if spec.horizon != expected_horizon:
        raise ValueError(
            f"Horizon invariant violation: timeframe '{spec.timeframe}' requires horizon={expected_horizon}, "
            f"got {spec.horizon}."
        )

    # 2. Alpha equals 2 * rank
    expected_alpha = 2 * spec.lora_r
    if spec.lora_alpha != expected_alpha:
        raise ValueError(
            f"Alpha invariant violation: rank={spec.lora_r} requires alpha={expected_alpha}, "
            f"got {spec.lora_alpha}."
        )

    # 3. Effective batch size equals 16
    if spec.effective_batch_size != 16:
        raise ValueError(
            f"Effective batch invariant violation: expected 16, got {spec.effective_batch_size} "
            f"({spec.batch_size} * {spec.gradient_accumulation_steps})."
        )

    # 4. Context length in allowed values
    if spec.context_len not in ALLOWED_CONTEXTS:
        raise ValueError(f"context_len {spec.context_len} not in {ALLOWED_CONTEXTS}.")

    # 5. Rank in allowed values
    if spec.lora_r not in ALLOWED_RANKS:
        raise ValueError(f"lora_r {spec.lora_r} not in {ALLOWED_RANKS}.")

    # 6. Feature set in allowed values
    if spec.feature_set not in ALLOWED_FEATURE_SETS:
        raise ValueError(f"feature_set {spec.feature_set} not in {ALLOWED_FEATURE_SETS}.")

    # 7. Learning rate within bounds
    if not (LR_MIN <= spec.learning_rate <= LR_MAX + 1e-9):
        raise ValueError(f"learning_rate {spec.learning_rate} out of bounds [{LR_MIN}, {LR_MAX}].")

    # 8. Max epochs capped at 10
    if spec.max_epochs > MAX_EPOCHS_AUTO:
        raise ValueError(f"max_epochs {spec.max_epochs} exceeds max limit {MAX_EPOCHS_AUTO}.")

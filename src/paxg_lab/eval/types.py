"""Data structures for forecasting requests, results, backtest specifications, and scores."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..constants import SCORE_VERSION, get_horizon_for_timeframe


@dataclass(frozen=True)
class ForecastRequest:
    symbol: str = "PAXGUSDT"
    timeframe: str = "1h"
    feature_set: str = "A"
    context_len: int = 256
    horizon: int = field(default=0)
    adapter_path: Path | str | None = None

    def __post_init__(self):
        expected_h = get_horizon_for_timeframe(self.timeframe)
        if self.horizon == 0:
            object.__setattr__(self, "horizon", expected_h)
        elif self.horizon != expected_h:
            raise ValueError(
                f"Invalid horizon {self.horizon} for timeframe {self.timeframe}. Expected {expected_h}."
            )


@dataclass(frozen=True)
class ForecastResult:
    symbol: str
    timeframe: str
    forecast_origin_time: int
    target_timestamps: list[int]
    point_forecast: np.ndarray  # shape: (horizon,) - median
    quantiles: np.ndarray  # shape: (horizon, 9) - monotonically sorted
    uncertainty_lower: np.ndarray  # shape: (horizon,) - q10
    uncertainty_upper: np.ndarray  # shape: (horizon,) - q90
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BacktestSpec:
    timeframe: str = "1h"
    feature_set: str = "A"
    context_len: int = 256
    step: int = 1
    batch_size: int = 16
    start_idx: int = 0
    end_idx: int | None = None


@dataclass(frozen=True)
class FoldMetrics:
    fold_id: int | str
    num_windows: int
    weighted_mae: float
    weighted_pinball: float
    rmse: float
    mae: float
    coverage_80: float
    mean_width_80: float
    directional_accuracy: float
    relative_mae_to_base: float = 1.0  # A_f
    relative_pinball_to_base: float = 1.0  # Q_f
    composite_loss: float = 1.0  # L_f = 0.70 A_f + 0.30 Q_f


@dataclass(frozen=True)
class ScoreReport:
    score_version: int = SCORE_VERSION
    timeframe: str = "1h"
    horizon: int = 24
    model_name: str = "TimesFM3-Base"
    feature_set: str = "A"
    context_len: int = 256
    score: float = 0.0  # 100 * [1 - (0.80 * mean(L_f) + 0.20 * max(L_f))]
    total_eval_windows: int = 0
    overall_mae: float = 0.0
    overall_rmse: float = 0.0
    overall_weighted_mae: float = 0.0
    overall_weighted_pinball: float = 0.0
    coverage_80: float = 0.0
    mean_width_80: float = 0.0
    directional_accuracy: float = 0.0
    step_mae: dict[int, float] = field(default_factory=dict)
    fold_metrics: list[FoldMetrics] = field(default_factory=list)
    test_metrics: FoldMetrics | None = None
    baseline_comparisons: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

"""Evaluation, forecasting predictor, and backtesting package for PAXG Forecast Lab."""

from .types import (
    ForecastRequest,
    ForecastResult,
    BacktestSpec,
    FoldMetrics,
    ScoreReport,
)
from .metrics import (
    calculate_weighted_mae,
    calculate_weighted_rmse,
    calculate_step_mae,
    calculate_weighted_pinball_loss,
    calculate_coverage_80,
    calculate_mean_width_80,
    calculate_directional_accuracy,
    compute_fold_metrics,
    compute_score_v1,
    calculate_low_move_breakdown,
    calculate_temporal_breakdown,
    calculate_volatility_breakdown,
    calculate_sampling_breakdown,
)
from .predictor import TimesFM3Predictor
from .engine import BacktestEngine

__all__ = [
    "ForecastRequest",
    "ForecastResult",
    "BacktestSpec",
    "FoldMetrics",
    "ScoreReport",
    "calculate_weighted_mae",
    "calculate_weighted_rmse",
    "calculate_step_mae",
    "calculate_weighted_pinball_loss",
    "calculate_coverage_80",
    "calculate_mean_width_80",
    "calculate_directional_accuracy",
    "calculate_low_move_breakdown",
    "calculate_temporal_breakdown",
    "calculate_volatility_breakdown",
    "calculate_sampling_breakdown",
    "compute_fold_metrics",
    "compute_score_v1",
    "TimesFM3Predictor",
    "BacktestEngine",
]

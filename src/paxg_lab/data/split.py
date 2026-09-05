"""Leak-free time series splitting and window slicing with strict horizon purge buffers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from ..constants import get_horizon_for_timeframe


@dataclass(frozen=True)
class FoldIndices:
    fold_id: int
    train_start: int
    train_end: int
    val_early_stop_start: int
    val_early_stop_end: int
    eval_start: int
    eval_end: int
    purge_buffer: int


@dataclass(frozen=True)
class SplitPlan:
    timeframe: str
    horizon: int
    total_candles: int
    train_history_end: int
    eval_folds: tuple[FoldIndices, ...]
    test_start: int
    test_end: int


def calculate_split_plan(
    total_candles: int,
    timeframe: str = "1h",
) -> SplitPlan:
    """Calculates leak-free temporal split boundaries according to plan specification.

    Specification:
      - 90 days test (locked verification)
      - 90 days prior split into 3 evaluation folds (30 days each)
      - Prior to each evaluation fold is training, with last 14 days for early stopping
      - Purge buffer between segments equals the timeframe horizon (24 for 1h, 6 for 4h)
    """
    horizon = get_horizon_for_timeframe(timeframe)
    candles_per_day = 24 if timeframe == "1h" else 6

    test_len = 90 * candles_per_day
    eval_fold_len = 30 * candles_per_day
    early_stop_len = 14 * candles_per_day
    total_eval_len = 3 * eval_fold_len  # 90 days

    min_required = test_len + total_eval_len + early_stop_len + 2 * horizon
    if total_candles < min_required:
        # If total history is shorter, scale proportionally while preserving rules
        scale_ratio = total_candles / min_required
        test_len = max(int(test_len * scale_ratio), 30 * candles_per_day)
        eval_fold_len = max(int(eval_fold_len * scale_ratio), 10 * candles_per_day)
        early_stop_len = max(int(early_stop_len * scale_ratio), 5 * candles_per_day)

    test_start = total_candles - test_len
    test_end = total_candles

    eval_total_start = test_start - (3 * eval_fold_len)

    folds = []
    for f in range(3):
        fold_eval_start = eval_total_start + f * eval_fold_len
        fold_eval_end = fold_eval_start + eval_fold_len

        # Training history is everything up to fold_eval_start - horizon (purge buffer)
        train_ceiling = fold_eval_start - horizon
        early_stop_start = max(train_ceiling - early_stop_len, 0)

        # Train end is before early stop
        train_start = 0
        train_end = early_stop_start

        folds.append(
            FoldIndices(
                fold_id=f + 1,
                train_start=train_start,
                train_end=train_end,
                val_early_stop_start=early_stop_start,
                val_early_stop_end=train_ceiling,
                eval_start=fold_eval_start,
                eval_end=fold_eval_end,
                purge_buffer=horizon,
            )
        )

    return SplitPlan(
        timeframe=timeframe,
        horizon=horizon,
        total_candles=total_candles,
        train_history_end=eval_total_start,
        eval_folds=tuple(folds),
        test_start=test_start,
        test_end=test_end,
    )


def extract_windows(
    features: np.ndarray,
    targets: np.ndarray,
    context_len: int,
    horizon: int,
    start_idx: int,
    end_idx: int,
    step: int = 1,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Extracts sliding context and future target windows strictly within [start_idx, end_idx].

    Args:
        features: (N, num_features) array.
        targets: (N,) or (N, 1) target array (close prices).
        context_len: Context length (e.g. 128, 256).
        horizon: Horizon length (24 for 1h, 6 for 4h).
        start_idx: Window start bound (context can look back before start_idx if known).
        end_idx: Strict upper bound: target future MUST NOT exceed end_idx.
        step: Stride for sliding window.

    Returns:
        Tuple of (context_windows, future_windows, forecast_origin_indices).
    """
    targets_1d = np.squeeze(targets)
    total_len = len(features)

    contexts = []
    futures = []
    origins = []

    # origin is the index of the last context point
    # target spans [origin + 1, origin + 1 + horizon]
    # condition: origin + 1 + horizon <= end_idx
    # and origin - context_len + 1 >= max(0, start_idx - context_len)
    min_origin = context_len - 1
    max_origin = end_idx - horizon - 1

    for origin in range(min_origin, max_origin + 1, step):
        ctx_slice = features[origin - context_len + 1 : origin + 1]
        fut_slice = targets_1d[origin + 1 : origin + 1 + horizon]

        if len(ctx_slice) == context_len and len(fut_slice) == horizon:
            contexts.append(ctx_slice)
            futures.append(fut_slice)
            origins.append(origin)

    if not contexts:
        return (
            np.empty((0, context_len, features.shape[-1]), dtype=features.dtype),
            np.empty((0, horizon), dtype=targets_1d.dtype),
            [],
        )

    return (
        np.array(contexts, dtype=features.dtype),
        np.array(futures, dtype=targets_1d.dtype),
        origins,
    )

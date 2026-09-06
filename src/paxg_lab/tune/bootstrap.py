"""Block Bootstrap and 24-hour independent block evaluation for time series verification.

Conforms strictly to PAXG Forecast Lab PLAN.md Section 3.5 & P6.md:
- Each independent block is exactly 24 hours:
  * 1h: 24 candles per block
  * 4h: 6 candles per block
- Minimum 20 non-overlapping blocks required for valid test verification.
- Block bootstrap (1000 iterations) calculates 95% confidence interval for MAE improvement:
  * Bootstrap CI must lie entirely on the better/positive side (CI_lower > 0).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from ..constants import get_horizon_for_timeframe, get_horizon_weights
from ..eval.metrics import calculate_weighted_mae


@dataclass(frozen=True)
class BlockBootstrapResult:
    """Statistical summary of block bootstrap evaluation on test verification set."""

    timeframe: str
    block_size_candles: int
    num_blocks: int
    candidate_weighted_mae: float
    baseline_weighted_mae: float
    mean_improvement_usdt: float
    relative_improvement_pct: float
    ci_95_lower: float
    ci_95_upper: float
    is_significant_positive: bool
    num_resamples: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def get_block_size_for_timeframe(timeframe: str) -> int:
    """Returns number of candles in one 24-hour block for the given timeframe.
    
    1h: 24 candles (24 * 1h = 24h)
    4h: 6 candles (6 * 4h = 24h)
    """
    tf = str(timeframe).lower().strip()
    if tf == "1h":
        return 24
    elif tf == "4h":
        return 6
    raise ValueError(f"Unsupported timeframe '{timeframe}'. Must be '1h' or '4h'.")


def compute_block_bootstrap_ci(
    candidate_predictions: np.ndarray,
    baseline_predictions: np.ndarray,
    targets: np.ndarray,
    timeframe: str,
    num_resamples: int = 1000,
    seed: int = 42,
    min_required_blocks: int = 20,
) -> BlockBootstrapResult:
    """Partitions test predictions into 24-hour non-overlapping blocks and computes 95% bootstrap CI.
    
    Args:
        candidate_predictions: (N, horizon) candidate point predictions.
        baseline_predictions: (N, horizon) baseline point predictions (e.g. base model or current recommended).
        targets: (N, horizon) actual future close prices.
        timeframe: "1h" or "4h".
        num_resamples: Number of bootstrap iterations (default 1000).
        seed: Random seed for bootstrap sampling.
        min_required_blocks: Minimum independent blocks required (default 20).
        
    Returns:
        BlockBootstrapResult containing block count, MAEs, 95% CI bounds, and significance verdict.
    """
    tf = str(timeframe).lower().strip()
    block_len = get_block_size_for_timeframe(tf)
    weights = get_horizon_weights(tf)

    num_windows = len(candidate_predictions)
    if num_windows != len(baseline_predictions) or num_windows != len(targets):
        raise ValueError(
            f"Shape mismatch: candidate ({len(candidate_predictions)}), "
            f"baseline ({len(baseline_predictions)}), targets ({len(targets)}) must have same length."
        )

    # In step=1 sliding windows, windows starting block_len apart are independent
    # A single window has horizon length = block_len (24 for 1h, 6 for 4h).
    # Therefore, taking non-overlapping window indices: origin, origin + block_len, origin + 2*block_len...
    num_blocks = num_windows // block_len
    if num_blocks < min_required_blocks:
        raise ValueError(
            f"Insufficient independent 24-hour blocks: got {num_blocks}, but minimum {min_required_blocks} "
            f"are required per PLAN 3.5 specification (need at least {min_required_blocks * block_len} test windows)."
        )

    # Calculate block-wise improvements across non-overlapping blocks
    block_improvements = []
    block_cand_maes = []
    block_base_maes = []

    for k in range(num_blocks):
        start = k * block_len
        end = start + block_len
        # Slice block windows
        c_block = candidate_predictions[start:end]
        b_block = baseline_predictions[start:end]
        t_block = targets[start:end]

        cand_mae = calculate_weighted_mae(c_block, t_block, weights)
        base_mae = calculate_weighted_mae(b_block, t_block, weights)

        # Improvement: positive means candidate has lower error than baseline
        improvement = base_mae - cand_mae

        block_cand_maes.append(cand_mae)
        block_base_maes.append(base_mae)
        block_improvements.append(improvement)

    arr_improvements = np.asarray(block_improvements, dtype=np.float64)
    overall_cand_mae = float(np.mean(block_cand_maes))
    overall_base_mae = float(np.mean(block_base_maes))
    mean_imp = float(np.mean(arr_improvements))
    rel_imp_pct = float(mean_imp / max(overall_base_mae, 1e-6) * 100.0)

    # Bootstrap resampling across blocks
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(num_resamples, dtype=np.float64)

    for b in range(num_resamples):
        sample_indices = rng.integers(0, num_blocks, size=num_blocks)
        bootstrap_means[b] = np.mean(arr_improvements[sample_indices])

    # 95% two-sided empirical percentile interval [2.5%, 97.5%]
    ci_lower = float(np.percentile(bootstrap_means, 2.5))
    ci_upper = float(np.percentile(bootstrap_means, 97.5))

    # Condition: 95% CI must lie entirely on the positive improvement side (ci_lower > 0)
    is_significant = bool(ci_lower > 0.0 and num_blocks >= min_required_blocks)

    return BlockBootstrapResult(
        timeframe=tf,
        block_size_candles=block_len,
        num_blocks=num_blocks,
        candidate_weighted_mae=overall_cand_mae,
        baseline_weighted_mae=overall_base_mae,
        mean_improvement_usdt=mean_imp,
        relative_improvement_pct=rel_imp_pct,
        ci_95_lower=ci_lower,
        ci_95_upper=ci_upper,
        is_significant_positive=is_significant,
        num_resamples=num_resamples,
    )

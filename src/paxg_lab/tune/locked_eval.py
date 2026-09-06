"""Locked test verification module for PAXG Forecast Lab Phase P6.

Executes single-pass, leak-free evaluation strictly on the 90-day locked test set [test_start, test_end).
Evaluates:
- Candidate model on locked test set
- Base TimesFM 3.0 model on locked test set
- Current Recommended adapter (if exists) on locked test set
- Naive flat baseline on locked test set
- 3 consecutive stability segments to ensure worst segment MAE <= 1.05 * Base
- Non-overlapping 24h independent blocks (origins spaced by horizon)
- 1000 resample Block Bootstrap 95% CI vs current recommended baseline (or Base)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
from pathlib import Path
from typing import Any

import numpy as np

from ..constants import get_horizon_for_timeframe, get_horizon_weights
from ..data.snapshot import DatasetSnapshot
from ..data.split import calculate_split_plan, extract_windows
from ..eval.metrics import (
    calculate_coverage_80,
    calculate_directional_accuracy,
    calculate_mean_width_80,
    calculate_step_mae,
    calculate_weighted_mae,
    calculate_weighted_pinball_loss,
    calculate_weighted_rmse,
    compute_score_v1,
)
from ..eval.predictor import TimesFM3Predictor
from ..model.manifest import AdapterManifest
from ..model.store import AdapterStore
from .bootstrap import BlockBootstrapResult, compute_block_bootstrap_ci, get_block_size_for_timeframe

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LockedVerificationReport:
    """Complete evaluation report strictly from the locked test set."""

    timeframe: str
    candidate_id: str
    test_start_idx: int
    test_end_idx: int
    score_v1: float
    current_recommended_score: float
    score_diff: float
    candidate_weighted_mae: float
    base_weighted_mae: float
    naive_weighted_mae: float
    baseline_weighted_mae: float
    baseline_name: str
    mae_vs_base_ratio: float
    mae_vs_naive_ratio: float
    segment_mae_ratios: dict[str, float]
    worst_segment_ratio: float
    coverage_80: float
    mean_width_80: float
    directional_accuracy: float
    independent_24h_blocks: int
    bootstrap_result: BlockBootstrapResult
    step_maes: dict[int, float]
    num_windows: int
    candidate_predictions: np.ndarray
    targets: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["candidate_predictions"] = self.candidate_predictions.tolist()
        d["targets"] = self.targets.tolist()
        return d


def run_locked_verification(
    snapshot: DatasetSnapshot,
    candidate_manifest: AdapterManifest,
    candidate_adapter_path: Path | str,
    store: AdapterStore | None = None,
    batch_size: int = 16,
) -> LockedVerificationReport:
    """Executes single-pass evaluation strictly on the locked test set [test_start, test_end).

    Args:
        snapshot: DatasetSnapshot containing full historical data.
        candidate_manifest: Manifest of the candidate adapter to evaluate.
        candidate_adapter_path: Path to the candidate adapter directory.
        store: AdapterStore instance to look up current recommended model.
        batch_size: Batch size for GPU inference.

    Returns:
        LockedVerificationReport containing all locked metrics required by Gatekeeper.
    """
    timeframe = str(candidate_manifest.timeframe).lower().strip()
    horizon = get_horizon_for_timeframe(timeframe)
    block_len = get_block_size_for_timeframe(timeframe)
    weights = get_horizon_weights(timeframe)

    split_plan = calculate_split_plan(
        total_candles=len(snapshot.features_a),
        timeframe=timeframe,
    )
    test_start = split_plan.test_start
    test_end = split_plan.test_end

    logger.info(
        "Starting locked test verification for %s [%d, %d) on candidate '%s'...",
        timeframe,
        test_start,
        test_end,
        candidate_manifest.adapter_id,
    )

    features = snapshot.get_features(candidate_manifest.feature_set)
    base_features = snapshot.get_features("A")
    targets = snapshot.features_a[:, 0]
    timestamps = snapshot.timestamps

    # 1. Partition locked test set into 3 consecutive stability segments
    total_test_candles = test_end - test_start
    seg_size = total_test_candles // 3
    segments = [
        ("seg1", test_start, test_start + seg_size),
        ("seg2", test_start + seg_size, test_start + 2 * seg_size),
        ("seg3", test_start + 2 * seg_size, test_end),
    ]

    # 2. Extract full test sliding windows
    cand_ctx, cand_fut, cand_origins = extract_windows(
        features=features,
        targets=targets,
        context_len=candidate_manifest.context_len,
        horizon=horizon,
        start_idx=test_start,
        end_idx=test_end,
        step=1,
        timestamps=timestamps,
        timeframe=timeframe,
    )
    base_ctx, base_fut, base_origins = extract_windows(
        features=base_features,
        targets=targets,
        context_len=256,
        horizon=horizon,
        start_idx=test_start,
        end_idx=test_end,
        step=1,
        timestamps=timestamps,
        timeframe=timeframe,
    )

    if len(cand_ctx) == 0:
        raise ValueError(f"No test windows could be extracted for range [{test_start}, {test_end}).")

    # 3. Single-pass candidate inference on locked test set
    cand_predictor = TimesFM3Predictor(adapter_path=candidate_adapter_path)
    cand_preds, cand_quantiles = cand_predictor.predict(
        contexts=cand_ctx,
        horizon=horizon,
        batch_size=batch_size,
    )

    # 4. Single-pass Base reference inference on locked test set
    base_predictor = TimesFM3Predictor()
    base_preds, base_quantiles = base_predictor.predict(
        contexts=base_ctx,
        horizon=horizon,
        batch_size=batch_size,
    )

    # 5. Naive flat baseline on locked test set
    origin_prices = np.asarray([targets[orig] for orig in cand_origins], dtype=np.float64)
    naive_preds = np.repeat(origin_prices[:, np.newaxis], horizon, axis=1)

    # 6. Evaluate Current Recommended adapter if one exists in store
    current_rec_id = store.get_recommended(timeframe) if store else None
    current_rec_score = 0.0
    rec_preds = None
    baseline_name = "TimesFM3-Base"

    if current_rec_id and store:
        try:
            rec_path = store.get_adapter_path(current_rec_id)
            rec_manifest = AdapterManifest.load_json(rec_path / "paxg_manifest.json")
            current_rec_score = float(rec_manifest.metrics.get("score_v1", 0.0))

            rec_feat = snapshot.get_features(rec_manifest.feature_set)
            rec_ctx, _, _ = extract_windows(
                features=rec_feat,
                targets=targets,
                context_len=rec_manifest.context_len,
                horizon=horizon,
                start_idx=test_start,
                end_idx=test_end,
                step=1,
                timestamps=timestamps,
                timeframe=timeframe,
            )
            rec_predictor = TimesFM3Predictor(adapter_path=rec_path)
            rec_preds, _ = rec_predictor.predict(
                contexts=rec_ctx,
                horizon=horizon,
                batch_size=batch_size,
            )
            baseline_name = current_rec_id
            logger.info("Loaded current recommended adapter '%s' for bootstrap baseline comparison.", current_rec_id)
        except Exception as exc:
            logger.warning("Could not evaluate current recommended adapter '%s': %s", current_rec_id, exc)
            rec_preds = None
            baseline_name = "TimesFM3-Base"

    bootstrap_baseline_preds = rec_preds if rec_preds is not None else base_preds

    # 7. Overall MAE and metric computations
    cand_weighted_mae = calculate_weighted_mae(cand_preds, cand_fut, weights)
    base_weighted_mae = calculate_weighted_mae(base_preds, cand_fut, weights)
    naive_weighted_mae = calculate_weighted_mae(naive_preds, cand_fut, weights)
    baseline_weighted_mae = calculate_weighted_mae(bootstrap_baseline_preds, cand_fut, weights)

    mae_vs_base_ratio = cand_weighted_mae / max(base_weighted_mae, 1e-6)
    mae_vs_naive_ratio = cand_weighted_mae / max(naive_weighted_mae, 1e-6)

    # 8. Stability segment analysis (3 segments) and composite Score v1
    segment_mae_ratios = {}
    segment_losses = []

    cand_origins_arr = np.asarray(cand_origins, dtype=np.int64)
    for seg_name, s_start, s_end in segments:
        seg_mask = (cand_origins_arr >= s_start) & (cand_origins_arr < s_end)
        if not np.any(seg_mask):
            continue
        c_seg_preds = cand_preds[seg_mask]
        b_seg_preds = base_preds[seg_mask]
        c_seg_quant = cand_quantiles[seg_mask]
        b_seg_quant = base_quantiles[seg_mask]
        t_seg_fut = cand_fut[seg_mask]

        c_seg_mae = calculate_weighted_mae(c_seg_preds, t_seg_fut, weights)
        b_seg_mae = calculate_weighted_mae(b_seg_preds, t_seg_fut, weights)
        c_seg_pinball = calculate_weighted_pinball_loss(c_seg_quant, t_seg_fut, weights)
        b_seg_pinball = calculate_weighted_pinball_loss(b_seg_quant, t_seg_fut, weights)

        ratio_mae = c_seg_mae / max(b_seg_mae, 1e-6)
        ratio_pinball = c_seg_pinball / max(b_seg_pinball, 1e-6)
        composite_l = 0.70 * ratio_mae + 0.30 * ratio_pinball

        segment_mae_ratios[seg_name] = float(ratio_mae)
        segment_losses.append(float(composite_l))

    worst_segment_ratio = max(segment_mae_ratios.values()) if segment_mae_ratios else 1.0
    locked_score_v1 = compute_score_v1(segment_losses) if segment_losses else 0.0
    score_diff = locked_score_v1 - current_rec_score

    # 9. Uncertainty coverage & step metrics
    cov_80 = calculate_coverage_80(cand_quantiles, cand_fut)
    width_80 = calculate_mean_width_80(cand_quantiles)
    dir_acc = calculate_directional_accuracy(cand_preds, cand_fut, origin_prices)
    step_maes = calculate_step_mae(cand_preds, cand_fut, timeframe=timeframe)

    # 10. Non-overlapping 24h independent blocks & Block Bootstrap CI
    bootstrap_res = compute_block_bootstrap_ci(
        candidate_predictions=cand_preds,
        baseline_predictions=bootstrap_baseline_preds,
        targets=cand_fut,
        timeframe=timeframe,
        baseline_name=baseline_name,
        num_resamples=1000,
        seed=42,
        min_required_blocks=20,
    )

    return LockedVerificationReport(
        timeframe=timeframe,
        candidate_id=candidate_manifest.adapter_id,
        test_start_idx=test_start,
        test_end_idx=test_end,
        score_v1=locked_score_v1,
        current_recommended_score=current_rec_score,
        score_diff=score_diff,
        candidate_weighted_mae=cand_weighted_mae,
        base_weighted_mae=base_weighted_mae,
        naive_weighted_mae=naive_weighted_mae,
        baseline_weighted_mae=baseline_weighted_mae,
        baseline_name=baseline_name,
        mae_vs_base_ratio=mae_vs_base_ratio,
        mae_vs_naive_ratio=mae_vs_naive_ratio,
        segment_mae_ratios=segment_mae_ratios,
        worst_segment_ratio=worst_segment_ratio,
        coverage_80=cov_80,
        mean_width_80=width_80,
        directional_accuracy=dir_acc,
        independent_24h_blocks=bootstrap_res.num_blocks,
        bootstrap_result=bootstrap_res,
        step_maes=step_maes,
        num_windows=len(cand_preds),
        candidate_predictions=cand_preds,
        targets=cand_fut,
    )

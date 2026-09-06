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
from typing import Any, Callable

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
    storage: Any | None = None,
    is_cancelled_func: Callable[[], bool] | None = None,
    split_plan: SplitPlan | None = None,
) -> LockedVerificationReport:
    """Executes single-pass evaluation strictly on the locked test set [test_start, test_end).

    Enforces:
    - Statistical lock: test range must not be re-evaluated if already consumed in ledger
    - Forecast origin alignment: all models evaluated strictly on common forecast origins
    - Contemporaneous baseline: current recommended adapter evaluated on the exact same locked set
    """
    timeframe = str(candidate_manifest.timeframe).lower().strip()
    horizon = get_horizon_for_timeframe(timeframe)
    block_len = get_block_size_for_timeframe(timeframe)
    weights = get_horizon_weights(timeframe)

    if split_plan is None:
        # Check if storage has persisted selected_test_range or prior consumed range
        custom_test_start = None
        if storage is not None and hasattr(storage, "get_auto_tune_run"):
            run_st = storage.get_auto_tune_run(timeframe)
            if run_st and run_st.get("selected_test_range_json"):
                try:
                    s_data = json.loads(run_st["selected_test_range_json"])
                    custom_test_start = s_data.get("test_start_idx")
                except Exception:
                    pass

        split_plan = calculate_split_plan(
            total_candles=len(snapshot.features_a),
            timeframe=timeframe,
            custom_test_start=custom_test_start,
        )
    test_start = split_plan.test_start
    test_end = split_plan.test_end
    snapshot_hash = ""
    if hasattr(snapshot, "metadata") and hasattr(snapshot.metadata, "sha256"):
        snapshot_hash = str(snapshot.metadata.sha256)

    if len(snapshot.timestamps) > 1:
        candle_step_ms = int(snapshot.timestamps[1] - snapshot.timestamps[0])
    else:
        from ..constants import timeframe_to_seconds
        candle_step_ms = int(timeframe_to_seconds(timeframe) * 1000)

    test_start_time_ms = int(snapshot.timestamps[test_start])
    if test_end < len(snapshot.timestamps):
        test_end_time_ms = int(snapshot.timestamps[test_end])
    else:
        test_end_time_ms = int(snapshot.timestamps[test_end - 1]) + candle_step_ms

    # 1. Enforce statistical lock via SQLite audit ledger
    if storage is not None:
        if storage.is_locked_range_consumed(
            timeframe=timeframe,
            test_start_time_ms=test_start_time_ms,
            test_end_time_ms=test_end_time_ms,
            snapshot_hash=snapshot_hash,
            test_start_idx=test_start,
            test_end_idx=test_end,
        ):
            raise RuntimeError(
                f"Statistical lock violation: locked verification interval [{test_start_time_ms}, {test_end_time_ms}) "
                f"for timeframe '{timeframe}' overlaps previously consumed test data in ledger. "
                "Refusing to reuse locked exam for another candidate."
            )
        # Atomically mark consumption before inference begins
        storage.record_locked_consumption(
            timeframe=timeframe,
            test_start_idx=test_start,
            test_end_idx=test_end,
            test_start_time_ms=test_start_time_ms,
            test_end_time_ms=test_end_time_ms,
            snapshot_hash=snapshot_hash,
            candidate_id=candidate_manifest.adapter_id,
            verdict="IN_PROGRESS",
            details="Locked verification initiated.",
        )

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

    # 2. Partition locked test set into 3 consecutive stability segments
    total_test_candles = test_end - test_start
    seg_size = total_test_candles // 3
    segments = [
        ("seg1", test_start, test_start + seg_size),
        ("seg2", test_start + seg_size, test_start + 2 * seg_size),
        ("seg3", test_start + 2 * seg_size, test_end),
    ]

    # 3. Extract candidate & base raw test sliding windows
    cand_ctx_raw, cand_fut_raw, cand_origins = extract_windows(
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
    base_ctx_raw, base_fut_raw, base_origins = extract_windows(
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

    # 4. Extract current recommended adapter windows if present in store
    current_rec_id = store.get_recommended(timeframe) if store else None
    rec_ctx_raw = None
    rec_fut_raw = None
    rec_origins = None
    rec_path = None
    if current_rec_id and store:
        try:
            rec_path = store.get_adapter_path(current_rec_id)
            rec_manifest = AdapterManifest.load_json(rec_path / "paxg_manifest.json")
            rec_feat = snapshot.get_features(rec_manifest.feature_set)
            rec_ctx_raw, rec_fut_raw, rec_origins = extract_windows(
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
        except Exception as exc:
            raise RuntimeError(
                f"Fail-closed: Failed preparing incumbent recommended adapter '{current_rec_id}' for contemporaneous evaluation: {exc}"
            ) from exc

    # 5. Intersect common forecast origins across candidate, base, and current recommended
    common_set = set(cand_origins) & set(base_origins)
    if rec_origins is not None:
        common_set = common_set & set(rec_origins)

    common_origins = sorted(common_set)
    if len(common_origins) == 0:
        raise ValueError(
            f"No common forecast origins across candidate, base, and recommended models in [{test_start}, {test_end})."
        )

    # Filter all context and target arrays strictly to common origins
    cand_orig_to_idx = {orig: i for i, orig in enumerate(cand_origins)}
    base_orig_to_idx = {orig: i for i, orig in enumerate(base_origins)}
    cand_indices = [cand_orig_to_idx[orig] for orig in common_origins]
    base_indices = [base_orig_to_idx[orig] for orig in common_origins]

    cand_ctx = cand_ctx_raw[cand_indices]
    cand_fut = cand_fut_raw[cand_indices]
    base_ctx = base_ctx_raw[base_indices]

    rec_ctx = None
    if rec_ctx_raw is not None and rec_origins is not None:
        rec_orig_to_idx = {orig: i for i, orig in enumerate(rec_origins)}
        rec_indices = [rec_orig_to_idx[orig] for orig in common_origins]
        rec_ctx = rec_ctx_raw[rec_indices]

    pred_kwargs: dict[str, Any] = {"batch_size": batch_size}
    if is_cancelled_func is not None:
        pred_kwargs["is_cancelled_func"] = is_cancelled_func

    # 6. Candidate inference
    if is_cancelled_func is not None and is_cancelled_func():
        raise InterruptedError("Locked verification cancelled by user request.")
    cand_predictor = TimesFM3Predictor(adapter_path=candidate_adapter_path)
    cand_preds, cand_quantiles = cand_predictor.predict(
        contexts=cand_ctx,
        horizon=horizon,
        **pred_kwargs,
    )

    # 7. Base reference inference
    if is_cancelled_func is not None and is_cancelled_func():
        raise InterruptedError("Locked verification cancelled by user request.")
    base_predictor = TimesFM3Predictor()
    base_preds, base_quantiles = base_predictor.predict(
        contexts=base_ctx,
        horizon=horizon,
        **pred_kwargs,
    )

    # 8. Current recommended inference (if exists)
    rec_preds = None
    rec_quantiles = None
    baseline_name = "TimesFM3-Base"
    if current_rec_id:
        if is_cancelled_func is not None and is_cancelled_func():
            raise InterruptedError("Locked verification cancelled by user request.")
        if rec_ctx is None or rec_path is None:
            raise RuntimeError(
                f"Fail-closed: Incumbent recommended adapter '{current_rec_id}' is active but context or adapter path is missing."
            )
        try:
            rec_predictor = TimesFM3Predictor(adapter_path=rec_path)
            rec_preds, rec_quantiles = rec_predictor.predict(
                contexts=rec_ctx,
                horizon=horizon,
                **pred_kwargs,
            )
            baseline_name = current_rec_id
            logger.info("Contemporaneous evaluation of recommended adapter '%s' on %d windows.", current_rec_id, len(rec_preds))
        except Exception as exc:
            raise RuntimeError(
                f"Fail-closed: Failed running inference on incumbent recommended adapter '{current_rec_id}': {exc}"
            ) from exc

    # 9. Naive flat baseline on common origins
    origin_prices = np.asarray([targets[orig] for orig in common_origins], dtype=np.float64)
    naive_preds = np.repeat(origin_prices[:, np.newaxis], horizon, axis=1)

    bootstrap_baseline_preds = rec_preds if rec_preds is not None else base_preds

    # 10. Overall MAE and metric computations
    cand_weighted_mae = calculate_weighted_mae(cand_preds, cand_fut, weights)
    base_weighted_mae = calculate_weighted_mae(base_preds, cand_fut, weights)
    naive_weighted_mae = calculate_weighted_mae(naive_preds, cand_fut, weights)
    baseline_weighted_mae = calculate_weighted_mae(bootstrap_baseline_preds, cand_fut, weights)

    mae_vs_base_ratio = cand_weighted_mae / max(base_weighted_mae, 1e-6)
    mae_vs_naive_ratio = cand_weighted_mae / max(naive_weighted_mae, 1e-6)

    # 11. Stability segment analysis (3 segments) & Contemporaneous Score v1
    segment_mae_ratios = {}
    segment_losses = []
    rec_segment_losses = []

    origins_arr = np.asarray(common_origins, dtype=np.int64)
    for seg_name, s_start, s_end in segments:
        seg_mask = (origins_arr >= s_start) & (origins_arr < s_end)
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

        # Contemporaneous composite loss for current recommended model on the same segment
        if rec_preds is not None and rec_quantiles is not None:
            r_seg_preds = rec_preds[seg_mask]
            r_seg_quant = rec_quantiles[seg_mask]
            r_seg_mae = calculate_weighted_mae(r_seg_preds, t_seg_fut, weights)
            r_seg_pinball = calculate_weighted_pinball_loss(r_seg_quant, t_seg_fut, weights)
            r_ratio_mae = r_seg_mae / max(b_seg_mae, 1e-6)
            r_ratio_pinball = r_seg_pinball / max(b_seg_pinball, 1e-6)
            r_composite_l = 0.70 * r_ratio_mae + 0.30 * r_ratio_pinball
            rec_segment_losses.append(float(r_composite_l))

    worst_segment_ratio = max(segment_mae_ratios.values()) if segment_mae_ratios else 1.0
    locked_score_v1 = compute_score_v1(segment_losses) if segment_losses else 0.0

    if rec_segment_losses:
        current_rec_score = compute_score_v1(rec_segment_losses)
    else:
        current_rec_score = 0.0

    score_diff = locked_score_v1 - current_rec_score

    # 12. Uncertainty coverage & step metrics
    cov_80 = calculate_coverage_80(cand_quantiles, cand_fut)
    width_80 = calculate_mean_width_80(cand_quantiles)
    dir_acc = calculate_directional_accuracy(cand_preds, cand_fut, origin_prices)
    step_maes = calculate_step_mae(cand_preds, cand_fut, timeframe=timeframe)

    # 13. Non-overlapping 24h independent blocks & Block Bootstrap CI
    if is_cancelled_func is not None and is_cancelled_func():
        raise InterruptedError("Locked verification cancelled by user request.")
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

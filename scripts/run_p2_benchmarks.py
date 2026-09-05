"""CLI script to run official Phase P2 benchmarks on 1h and 4h snapshots."""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

# Safe encoding for Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import torch

from paxg_lab.constants import (
    MODEL_REPO,
    MODEL_REVISION,
    SCORE_VERSION,
    TIMEFRAME_HORIZONS,
)
from paxg_lab.data.snapshot import DatasetSnapshot, SNAPSHOTS_DIR
from paxg_lab.eval import BacktestEngine, TimesFM3Predictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def find_latest_snapshot(timeframe: str) -> DatasetSnapshot:
    """Finds and loads the latest snapshot for the timeframe."""
    candidates = sorted(list(SNAPSHOTS_DIR.glob(f"paxgusdt_{timeframe}_*")))
    if not candidates:
        raise FileNotFoundError(f"No snapshot found for timeframe '{timeframe}' in {SNAPSHOTS_DIR}")
    latest_dir = candidates[-1]
    logger.info("Loading snapshot for %s from %s...", timeframe, latest_dir)
    return DatasetSnapshot.load(latest_dir, verify_hash=True)


def main() -> None:
    print("=" * 80)
    print("  PAXG FORECAST LAB — PHASE P2 BASE TIMESFM 3.0 BENCHMARKS")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"  Device:         {device} ({device_name})")
    print(f"  PyTorch:        {torch.__version__}")
    print(f"  Model:          {MODEL_REPO} (rev: {MODEL_REVISION[:8]})")
    print(f"  Horizons:       1h -> {TIMEFRAME_HORIZONS['1h']} steps; 4h -> {TIMEFRAME_HORIZONS['4h']} steps")
    print("=" * 80)

    # 1. Load Snapshots
    snap_1h = find_latest_snapshot("1h")
    snap_4h = find_latest_snapshot("4h")

    # 2. Initialize Predictor and Engine
    predictor = TimesFM3Predictor(device=device)
    engine = BacktestEngine(predictor=predictor)

    # 3. Run 1h Backtest (eval folds only, test set locked per PLAN)
    print("\n>>> Running 1h Base Benchmark (horizon = 24 steps = 24 hours, eval folds only)...")
    t0_1h = time.time()
    rep_1h = engine.run_full_backtest(
        snapshot=snap_1h,
        feature_set="A",
        context_len=256,
        batch_size=32 if device == "cuda" else 8,
        model_name="TimesFM3-Base",
        is_base_reference=True,
        include_locked_test=False,
    )
    elapsed_1h = time.time() - t0_1h
    print(f"    Completed 1h in {elapsed_1h:.1f}s ({rep_1h.total_eval_windows} eval windows)")

    # 4. Run 4h Backtest (eval folds only, test set locked per PLAN)
    print("\n>>> Running 4h Base Benchmark (horizon = 6 steps = 24 hours, eval folds only)...")
    t0_4h = time.time()
    rep_4h = engine.run_full_backtest(
        snapshot=snap_4h,
        feature_set="A",
        context_len=256,
        batch_size=32 if device == "cuda" else 8,
        model_name="TimesFM3-Base",
        is_base_reference=True,
        include_locked_test=False,
    )
    elapsed_4h = time.time() - t0_4h
    print(f"    Completed 4h in {elapsed_4h:.1f}s ({rep_4h.total_eval_windows} eval windows)")

    # 5. Summary Display
    print("\n" + "=" * 80)
    print("  PHASE P2 BENCHMARK RESULTS SUMMARY (EVALUATION FOLDS ONLY — TEST SET LOCKED)")
    print("=" * 80)

    print("\n[1h Benchmark - 24 steps]")
    print(f"  Score v1:               {rep_1h.score:.2f} (Base reference standard = 0.0)")
    print(f"  Weighted MAE:           ${rep_1h.overall_weighted_mae:.2f} USDT")
    print(f"  Weighted RMSE:          ${rep_1h.overall_rmse:.2f} USDT")
    print(f"  80% Uncertainty Cov:    {rep_1h.coverage_80 * 100:.1f}%")
    print(f"  80% Mean Width:         ${rep_1h.mean_width_80:.2f} USDT")
    print(f"  Directional Accuracy:   {rep_1h.directional_accuracy * 100:.1f}%")
    print("  Milestone Step MAEs:")
    for step, val in rep_1h.step_mae.items():
        print(f"    Step {step:2d} (+{step}h):        ${val:.2f} USDT")
    print(f"  Naive Flat MAE:         ${rep_1h.baseline_comparisons['naive_flat_eval_weighted_mae']:.2f} USDT")
    print(f"  Improvement over Naive: {rep_1h.baseline_comparisons['improvement_over_naive_pct']:.2f}%")

    print("\n[4h Benchmark - 6 steps]")
    print(f"  Score v1:               {rep_4h.score:.2f} (Base reference standard = 0.0)")
    print(f"  Weighted MAE:           ${rep_4h.overall_weighted_mae:.2f} USDT")
    print(f"  Weighted RMSE:          ${rep_4h.overall_rmse:.2f} USDT")
    print(f"  80% Uncertainty Cov:    {rep_4h.coverage_80 * 100:.1f}%")
    print(f"  80% Mean Width:         ${rep_4h.mean_width_80:.2f} USDT")
    print(f"  Directional Accuracy:   {rep_4h.directional_accuracy * 100:.1f}%")
    print("  Milestone Step MAEs:")
    for step, val in rep_4h.step_mae.items():
        print(f"    Step {step:2d} (+{step * 4}h):       ${val:.2f} USDT")
    print(f"  Naive Flat MAE:         ${rep_4h.baseline_comparisons['naive_flat_eval_weighted_mae']:.2f} USDT")
    print(f"  Improvement over Naive: {rep_4h.baseline_comparisons['improvement_over_naive_pct']:.2f}%")

    # 6. Save Evidence JSON (Eval folds only, zero test set contamination)
    evidence = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "policy": "EVAL_FOLDS_ONLY_LOCKED_TEST_PROTECTED",
        "hardware": {
            "device": str(device),
            "device_name": device_name,
            "pytorch_version": torch.__version__,
        },
        "model": {
            "repo": MODEL_REPO,
            "revision": MODEL_REVISION,
            "architecture": "TimesFM 3.0 PyTorch Base",
        },
        "benchmarks": {
            "1h": {
                "score_version": rep_1h.score_version,
                "score_v1": rep_1h.score,
                "horizon_steps": rep_1h.horizon,
                "eval_windows": rep_1h.total_eval_windows,
                "weighted_mae": rep_1h.overall_weighted_mae,
                "weighted_rmse": rep_1h.overall_rmse,
                "raw_mae": rep_1h.overall_mae,
                "coverage_80_pct": rep_1h.coverage_80 * 100.0,
                "mean_width_80_usdt": rep_1h.mean_width_80,
                "directional_accuracy_pct": rep_1h.directional_accuracy * 100.0,
                "step_maes_usdt": rep_1h.step_mae,
                "naive_comparison": {
                    "naive_flat_eval_weighted_mae": rep_1h.baseline_comparisons["naive_flat_eval_weighted_mae"],
                    "improvement_over_naive_pct": rep_1h.baseline_comparisons["improvement_over_naive_pct"],
                },
                "breakdowns": rep_1h.breakdowns,
                "folds": [
                    {
                        "fold_id": f.fold_id,
                        "windows": f.num_windows,
                        "weighted_mae": f.weighted_mae,
                        "weighted_pinball": f.weighted_pinball,
                        "relative_mae_to_base": f.relative_mae_to_base,
                        "relative_pinball_to_base": f.relative_pinball_to_base,
                        "composite_loss": f.composite_loss,
                        "reference_valid": f.reference_valid,
                    }
                    for f in rep_1h.fold_metrics
                ],
                "locked_test_status": "LOCKED_UNTOUCHED",
            },
            "4h": {
                "score_version": rep_4h.score_version,
                "score_v1": rep_4h.score,
                "horizon_steps": rep_4h.horizon,
                "eval_windows": rep_4h.total_eval_windows,
                "weighted_mae": rep_4h.overall_weighted_mae,
                "weighted_rmse": rep_4h.overall_rmse,
                "raw_mae": rep_4h.overall_mae,
                "coverage_80_pct": rep_4h.coverage_80 * 100.0,
                "mean_width_80_usdt": rep_4h.mean_width_80,
                "directional_accuracy_pct": rep_4h.directional_accuracy * 100.0,
                "step_maes_usdt": rep_4h.step_mae,
                "naive_comparison": {
                    "naive_flat_eval_weighted_mae": rep_4h.baseline_comparisons["naive_flat_eval_weighted_mae"],
                    "improvement_over_naive_pct": rep_4h.baseline_comparisons["improvement_over_naive_pct"],
                },
                "breakdowns": rep_4h.breakdowns,
                "folds": [
                    {
                        "fold_id": f.fold_id,
                        "windows": f.num_windows,
                        "weighted_mae": f.weighted_mae,
                        "weighted_pinball": f.weighted_pinball,
                        "relative_mae_to_base": f.relative_mae_to_base,
                        "relative_pinball_to_base": f.relative_pinball_to_base,
                        "composite_loss": f.composite_loss,
                        "reference_valid": f.reference_valid,
                    }
                    for f in rep_4h.fold_metrics
                ],
                "locked_test_status": "LOCKED_UNTOUCHED",
            },
        },
    }

    evidence_path = Path("docs/paxg-lab/phases/p2_base_benchmark.json")
    with open(evidence_path, "w", encoding="utf-8") as f:
        json.dump(evidence, f, indent=2, ensure_ascii=False)

    print(f"\nSaved benchmark evidence to {evidence_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()

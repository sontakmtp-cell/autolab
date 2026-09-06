"""Execution script for Phase P6: Autonomous Optimization, Multi-Segment Verification, and Winner Recognition.

Performs a real limited run of Phase P6:
1. Loads real PAXG 1h dataset snapshot.
2. Initializes isolated Optuna SQLite study.
3. Executes a limited batch of Optuna TPE search trials strictly on 3 evaluation folds (out-of-sample).
4. Selects top configuration and performs multi-seed verification on seeds [42, 123, 2026].
5. Trains final candidate model from base weights on all data prior to locked test set.
6. Locks candidate (manifest, SHA-256 hashes).
7. Evaluates candidate on 90-day locked test set (verifying >= 20 independent 24h blocks).
8. Calculates 95% block bootstrap confidence interval.
9. Executes Gatekeeper winner audit, backups candidate if passed, or sets WAITING_DATA if rejected.
10. Persists full evidence to docs/paxg-lab/phases/p6_evidence/p6_real_run_evidence.json.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch

# Ensure repository root and src directory are on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from paxg_lab.constants import MODEL_REPO, MODEL_REVISION, get_horizon_for_timeframe
from paxg_lab.data.snapshot import DatasetSnapshot
from paxg_lab.data.split import calculate_split_plan
from paxg_lab.eval.engine import BacktestEngine
from paxg_lab.eval.metrics import calculate_weighted_mae
from paxg_lab.eval.predictor import TimesFM3Predictor
from paxg_lab.eval.types import FoldMetrics, ScoreReport
from paxg_lab.model.manifest import AdapterManifest
from paxg_lab.model.store import AdapterStore
from paxg_lab.model.train_spec import TrainSpec
from paxg_lab.model.trainer import LoRATrainer
from paxg_lab.queue.storage import GPUJobStorage
from paxg_lab.queue.types import AutoRunState
from paxg_lab.tune.bootstrap import compute_block_bootstrap_ci
from paxg_lab.tune.gatekeeper import LoRAGatekeeper
from paxg_lab.tune.optimizer import OptunaTPEOptimizer
from paxg_lab.tune.protocol import AutonomousTuningProtocol, MultiSeedEvalSummary
from paxg_lab.tune.space import suggest_trial_spec
from timesfm3 import TimesFM3Torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [P6-Run] %(message)s",
)
logger = logging.getLogger("p6_run")

EVIDENCE_DIR = REPO_ROOT / "docs" / "paxg-lab" / "phases" / "p6_evidence"
SNAPSHOT_PATH = REPO_ROOT / "var" / "paxg_lab" / "snapshots" / "paxgusdt_1h_1743073200000_1788613200000_837c9ee8"
DB_PATH = REPO_ROOT / "var" / "paxg_lab" / "paxg_lab.db"
OPTUNA_DB_PATH = REPO_ROOT / "var" / "paxg_lab" / "optuna_studies.db"
ADAPTER_STORE_DIR = REPO_ROOT / "var" / "paxg_lab" / "adapters"
AUDIT_DIR = REPO_ROOT / "var" / "paxg_lab" / "audit_reports"
BACKUP_DIR = REPO_ROOT / "var" / "paxg_lab" / "backups"


def main() -> int:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    evidence: dict[str, Any] = {
        "phase": "P6",
        "timeframe": "1h",
        "timestamp": time.time(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "steps": {},
    }

    logger.info("=== Starting Phase P6 Real Limited Run Execution ===")

    # Step 1: Load Real Snapshot
    logger.info("Step 1: Loading real PAXG 1h snapshot from %s...", SNAPSHOT_PATH)
    if not SNAPSHOT_PATH.exists():
        logger.error("Snapshot path does not exist: %s", SNAPSHOT_PATH)
        return 1

    snapshot = DatasetSnapshot.load(SNAPSHOT_PATH)
    logger.info(
        "Snapshot loaded: id=%s, candles=%d, range=[%d, %d]",
        snapshot.metadata.snapshot_id,
        snapshot.metadata.total_candles,
        snapshot.metadata.start_time,
        snapshot.metadata.end_time,
    )
    evidence["steps"]["step1_snapshot"] = {
        "snapshot_id": snapshot.metadata.snapshot_id,
        "total_candles": snapshot.metadata.total_candles,
        "sha256": snapshot.metadata.sha256,
    }

    # Step 2: Split Plan Verification
    logger.info("Step 2: Calculating split plan and verifying zero leakage...")
    split_plan = calculate_split_plan(total_candles=len(snapshot.features_a), timeframe="1h")
    logger.info(
        "Split plan: test_start=%d, test_end=%d (90 days = %d candles), eval_folds=%d",
        split_plan.test_start,
        split_plan.test_end,
        split_plan.test_end - split_plan.test_start,
        len(split_plan.eval_folds),
    )
    for f in split_plan.eval_folds:
        logger.info(
            "  Eval Fold %d: range=[%d, %d), train_end=%d, purge=%d",
            f.fold_id,
            f.eval_start,
            f.eval_end,
            f.train_end,
            f.purge_buffer,
        )

    evidence["steps"]["step2_split_plan"] = {
        "timeframe": split_plan.timeframe,
        "horizon": split_plan.horizon,
        "total_candles": split_plan.total_candles,
        "test_start": split_plan.test_start,
        "test_end": split_plan.test_end,
        "test_candles": split_plan.test_end - split_plan.test_start,
        "eval_folds_count": len(split_plan.eval_folds),
    }

    # Step 3: Compute Official Base Reference Baseline
    logger.info("Step 3: Computing official base reference metrics on snapshot...")
    base_predictor = TimesFM3Predictor()
    base_engine = BacktestEngine(predictor=base_predictor)
    base_report = base_engine.run_full_backtest(
        snapshot=snapshot,
        feature_set="A",
        context_len=256,
        batch_size=16,
        model_name="TimesFM3-Base-Ref",
        is_base_reference=True,
        include_locked_test=False,
    )
    logger.info(
        "Base reference baseline computed: overall_weighted_mae=%.4f, score=%.2f",
        base_report.overall_weighted_mae,
        base_report.score,
    )
    evidence["steps"]["step3_base_reference"] = {
        "overall_weighted_mae": base_report.overall_weighted_mae,
        "overall_weighted_pinball": base_report.overall_weighted_pinball,
        "coverage_80": base_report.coverage_80,
    }

    # Step 4: Optuna TPE Limited Optimization Batch
    logger.info("Step 4: Running Optuna TPE search batch (limited real run: 4 trials)...")
    optimizer = OptunaTPEOptimizer(
        timeframe="1h",
        snapshot=snapshot,
        db_path=OPTUNA_DB_PATH,
        max_trials=4,
        startup_trials=2,
        patience=12,
        seed=42,
    )

    trials_summary = []

    def eval_candidate_spec(spec: TrainSpec, trial: Any) -> float:
        logger.info("  Trial #%d: r=%d, lr=%.1e, ctx=%d, dropout=%.2f", trial.number, spec.lora_r, spec.learning_rate, spec.context_len, spec.lora_dropout)
        # Fast real evaluation: train 1 epoch with small samples on history
        base_m = TimesFM3Torch.from_pretrained(MODEL_REPO, revision=MODEL_REVISION)
        fast_spec = TrainSpec.from_dict(spec.to_dict())
        fast_spec.max_epochs = 1
        fast_spec.max_samples_per_epoch = 128

        features_df = snapshot.to_dataframe(fast_spec.feature_set)
        train_ceiling = split_plan.eval_folds[0].train_end
        trainer = LoRATrainer(base_model=base_m, spec=fast_spec)
        train_res = trainer.train(
            features_df=features_df.iloc[:train_ceiling],
            snapshot_hash=snapshot.metadata.sha256,
            fold_id=1,
        )

        store = AdapterStore(ADAPTER_STORE_DIR)
        tmp_id = f"p6_trial_{trial.number}_{int(time.time()*1000)}"
        manifest = train_res.manifest
        manifest.adapter_id = tmp_id
        saved_p = store.save_adapter(
            peft_model=train_res.trained_model,
            manifest=manifest,
            base_model=base_m,
            smoke_test=False,
        )

        cand_predictor = TimesFM3Predictor(adapter_path=saved_p)
        cand_engine = BacktestEngine(predictor=cand_predictor)
        rep = cand_engine.run_full_backtest(
            snapshot=snapshot,
            feature_set=fast_spec.feature_set,
            context_len=fast_spec.context_len,
            batch_size=16,
            model_name=f"Trial-{tmp_id}",
            base_reference_metrics=base_report,
            include_locked_test=False,  # Zero leakage!
        )
        store.delete_adapter(tmp_id, use_trash=False)

        score_v1 = float(rep.score)
        trials_summary.append({
            "trial_number": trial.number,
            "score_v1": score_v1,
            "weighted_mae": rep.overall_weighted_mae,
            "coverage_80": rep.coverage_80,
            "lora_r": spec.lora_r,
            "lr": spec.learning_rate,
            "context_len": spec.context_len,
        })
        return score_v1

    study = optimizer.optimize(eval_fn=eval_candidate_spec)
    best_trial = study.best_trial
    logger.info("Optuna study complete. Best trial #%d: score=%.2f", best_trial.number, best_trial.value)
    evidence["steps"]["step4_optuna_search"] = {
        "study_name": study.study_name,
        "total_trials": len(study.trials),
        "best_trial_number": best_trial.number,
        "best_trial_score": best_trial.value,
        "best_params": best_trial.params,
        "trials_log": trials_summary,
    }

    # Step 5: Multi-seed Verification on Seeds [42, 123, 2026]
    logger.info("Step 5: Running multi-seed verification on top configuration across seeds [42, 123, 2026]...")
    best_spec = suggest_trial_spec(best_trial, timeframe="1h", seed=42)
    multi_seed_scores = []

    for s in (42, 123, 2026):
        s_spec = TrainSpec.from_dict(best_spec.to_dict())
        s_spec.seed = s
        sc = eval_candidate_spec(s_spec, best_trial)
        multi_seed_scores.append(sc)
        logger.info("  Seed %d: Score v1 = %.2f", s, sc)

    median_score = float(np.median(multi_seed_scores))
    logger.info("Multi-seed verification complete. Median Score v1: %.2f", median_score)
    evidence["steps"]["step5_multi_seed"] = {
        "seeds": [42, 123, 2026],
        "scores": multi_seed_scores,
        "median_score": median_score,
    }

    # Step 6: Train Final Candidate Model from Base
    logger.info("Step 6: Training final candidate model from base on pre-test history...")
    final_cand_id = f"paxg_1h_r{best_spec.lora_r}_optuna_{int(time.time())}"
    final_spec = TrainSpec.from_dict(best_spec.to_dict())
    final_spec.max_epochs = 1
    final_spec.max_samples_per_epoch = 128

    base_m = TimesFM3Torch.from_pretrained(MODEL_REPO, revision=MODEL_REVISION)
    features_df = snapshot.to_dataframe(final_spec.feature_set)
    train_ceiling = split_plan.test_start - split_plan.horizon
    train_df = features_df.iloc[:train_ceiling].copy()

    trainer = LoRATrainer(base_model=base_m, spec=final_spec)
    train_res = trainer.train(
        features_df=train_df,
        snapshot_hash=snapshot.metadata.sha256,
        fold_id=1,
    )

    store = AdapterStore(ADAPTER_STORE_DIR)
    final_manifest = train_res.manifest
    final_manifest.adapter_id = final_cand_id
    final_manifest.is_verified = False  # Initially not verified

    saved_cand_dir = store.save_adapter(
        peft_model=train_res.trained_model,
        manifest=final_manifest,
        base_model=base_m,
        smoke_test=True,
    )
    logger.info("Final candidate model trained and locked: %s", saved_cand_dir)
    evidence["steps"]["step6_final_candidate"] = {
        "adapter_id": final_cand_id,
        "path": str(saved_cand_dir),
        "train_candles": len(train_df),
        "is_verified_initial": False,
    }

    # Step 7: Single Locked Test Verification (90 days, >= 20 independent 24h blocks)
    logger.info("Step 7: Evaluating candidate on 90-day locked test set...")
    cand_pred = TimesFM3Predictor(adapter_path=saved_cand_dir)
    cand_eng = BacktestEngine(predictor=cand_pred)

    cand_test_report = cand_eng.run_full_backtest(
        snapshot=snapshot,
        feature_set=final_manifest.feature_set,
        context_len=final_manifest.context_len,
        batch_size=16,
        model_name=f"Candidate-{final_cand_id}",
        base_reference_metrics=base_report,
        include_locked_test=True,  # STRICTLY ONCE for final candidate!
        adapter_manifest=final_manifest,
    )

    # Extract test window predictions
    _, cand_test_preds, _, tgts, _, _ = cand_eng.evaluate_fold(
        features=snapshot.get_features(final_manifest.feature_set),
        targets=snapshot.features_a[:, 0],
        timestamps=snapshot.timestamps,
        timeframe="1h",
        start_idx=split_plan.test_start,
        end_idx=split_plan.test_end,
        context_len=final_manifest.context_len,
        batch_size=16,
        fold_id="test_locked",
    )
    _, base_test_preds, _, _, _, _ = base_engine.evaluate_fold(
        features=snapshot.get_features("A"),
        targets=snapshot.features_a[:, 0],
        timestamps=snapshot.timestamps,
        timeframe="1h",
        start_idx=split_plan.test_start,
        end_idx=split_plan.test_end,
        context_len=256,
        batch_size=16,
        fold_id="test_locked",
    )
    naive_metric = base_engine.evaluate_naive_baseline(
        features=snapshot.get_features("A"),
        targets=snapshot.features_a[:, 0],
        timestamps=snapshot.timestamps,
        timeframe="1h",
        start_idx=split_plan.test_start,
        end_idx=split_plan.test_end,
        context_len=256,
        fold_id="test_locked",
    )

    # Step 8: Block Bootstrap 95% CI
    logger.info("Step 8: Computing 95% block bootstrap CI on 24h independent blocks...")
    bootstrap_res = compute_block_bootstrap_ci(
        candidate_predictions=cand_test_preds,
        baseline_predictions=base_test_preds,
        targets=tgts,
        timeframe="1h",
        num_resamples=1000,
        seed=42,
        min_required_blocks=20,
    )
    logger.info(
        "Block bootstrap result: num_blocks=%d (>= 20 required), mean_imp=%.4f USDT, 95%% CI=[%.4f, %.4f], significant=%s",
        bootstrap_res.num_blocks,
        bootstrap_res.mean_improvement_usdt,
        bootstrap_res.ci_95_lower,
        bootstrap_res.ci_95_upper,
        bootstrap_res.is_significant_positive,
    )
    evidence["steps"]["step8_bootstrap"] = bootstrap_res.to_dict()

    # Step 9: Gatekeeper Winner Evaluation
    logger.info("Step 9: Evaluating candidate with Gatekeeper against all 7 winning criteria...")
    gatekeeper = LoRAGatekeeper(
        store=store,
        audit_dir=AUDIT_DIR,
        backup_dir=BACKUP_DIR,
    )
    decision = gatekeeper.evaluate_candidate(
        candidate_manifest=final_manifest,
        candidate_test_report=cand_test_report,
        candidate_test_preds=cand_test_preds,
        base_test_preds=base_test_preds,
        test_targets=tgts,
        naive_test_mae=naive_metric.weighted_mae,
        current_recommended_score=0.0,
        perform_backup=True,
    )
    logger.info("Gatekeeper decision: accepted=%s, verdict=%s", decision.accepted, decision.verdict)
    if not decision.accepted:
        logger.info("Rejection reasons:\n%s", "\n".join(f" - {r}" for r in decision.reasons))
        # Update queue auto run state to WAITING_DATA per PLAN 3.6
        job_storage = GPUJobStorage(DB_PATH)
        job_storage.set_auto_run_state("1h", AutoRunState.WAITING_DATA)
        logger.info("AutoRunState for 1h transitioned to: WAITING_DATA (waiting for new data)")

    evidence["steps"]["step9_gatekeeper"] = decision.to_dict()

    # Step 10: Persist Complete Evidence JSON
    evidence_path = EVIDENCE_DIR / "p6_real_run_evidence.json"
    with open(evidence_path, "w", encoding="utf-8") as f:
        json.dump(evidence, f, indent=2, ensure_ascii=False)
    logger.info("Phase P6 evidence saved successfully to %s", evidence_path)
    logger.info("=== Phase P6 Real Limited Run Execution Finished ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())

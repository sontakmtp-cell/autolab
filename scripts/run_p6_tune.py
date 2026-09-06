"""Execution script for Phase P6: Autonomous Optimization, Multi-Segment Verification, and Winner Recognition.

Delegates to AutonomousTuningProtocol.run_tuning_cycle() which orchestrates:
1. Loads real PAXG 1h dataset snapshot.
2. Bayesian hyperparameter search via Optuna TPE (limited: 4 trials for evidence).
3. Top-3 unique config multi-seed verification on seeds [42, 123, 2026].
4. Final candidate training from base weights on all history prior to locked test set.
5. Candidate lock (manifest, SHA-256 checksums).
6. Single locked verification on 90-day test set (>= 20 independent 24h blocks).
7. Block bootstrap 95% CI + Gatekeeper winner evaluation.
8. Persists full evidence to docs/paxg-lab/phases/p6_evidence/p6_real_run_evidence.json.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
import time
from typing import Any

import torch

# Ensure repository root and src directory are on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from paxg_lab.data.snapshot import DatasetSnapshot
from paxg_lab.data.split import calculate_split_plan
from paxg_lab.tune.protocol import AutonomousTuningProtocol

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

MAX_TRIALS = 4


def main() -> int:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=== Starting Phase P6 Real Limited Run Execution ===")

    # ── Step 1: Load Real Snapshot ──────────────────────────────────────────
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

    # ── Step 2: Log Split Plan ──────────────────────────────────────────────
    split_plan = calculate_split_plan(total_candles=len(snapshot.features_a), timeframe="1h")
    logger.info(
        "Split plan: test_start=%d, test_end=%d (%d candles), eval_folds=%d",
        split_plan.test_start,
        split_plan.test_end,
        split_plan.test_end - split_plan.test_start,
        len(split_plan.eval_folds),
    )

    # ── Step 3–9: Delegate to AutonomousTuningProtocol ──────────────────────
    logger.info("Delegating to AutonomousTuningProtocol.run_tuning_cycle(max_trials=%d)...", MAX_TRIALS)
    protocol = AutonomousTuningProtocol(
        timeframe="1h",
        snapshot=snapshot,
        db_path=DB_PATH,
        optuna_db_path=OPTUNA_DB_PATH,
        adapter_store_dir=ADAPTER_STORE_DIR,
        audit_dir=AUDIT_DIR,
        backup_dir=BACKUP_DIR,
        max_trials=MAX_TRIALS,
        startup_trials=2,
        patience=12,
    )

    result = protocol.run_tuning_cycle(max_trials=MAX_TRIALS)
    result_dict = result.to_dict()
    logger.info("Tuning cycle finished. Final state: %s", result.resulting_auto_state)

    # ── Build Evidence JSON ─────────────────────────────────────────────────
    evidence: dict[str, Any] = {
        "phase": "P6",
        "timeframe": "1h",
        "timestamp": time.time(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "steps": {
            "step1_snapshot": {
                "snapshot_id": snapshot.metadata.snapshot_id,
                "total_candles": snapshot.metadata.total_candles,
                "sha256": snapshot.metadata.sha256,
            },
            "step2_split_plan": {
                "timeframe": split_plan.timeframe,
                "horizon": split_plan.horizon,
                "total_candles": split_plan.total_candles,
                "test_start": split_plan.test_start,
                "test_end": split_plan.test_end,
                "test_candles": split_plan.test_end - split_plan.test_start,
                "eval_folds_count": len(split_plan.eval_folds),
            },
            "step3_protocol_run": {
                "study_name": result.study_name,
                "total_trials": result.total_trials,
                "best_trial_number": result.best_trial_number,
                "best_trial_params": result.best_trial_params,
                "multi_seed_summary": result.multi_seed_summary,
                "final_candidate_id": result.final_candidate_id,
                "test_score_report": result.test_score_report,
                "decision": result.decision,
                "resulting_auto_state": result.resulting_auto_state,
                "finished_at": result.finished_at,
            },
        },
    }

    # ── Persist Evidence ────────────────────────────────────────────────────
    evidence_path = EVIDENCE_DIR / "p6_real_run_evidence.json"
    with open(evidence_path, "w", encoding="utf-8") as f:
        json.dump(evidence, f, indent=2, ensure_ascii=False)
    logger.info("Phase P6 evidence saved to %s", evidence_path)
    logger.info("=== Phase P6 Real Limited Run Execution Finished ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Strict gatekeeper verifying candidate adapters and enforcing winner recognition rules.

Conforms strictly to PAXG Forecast Lab PLAN.md Section 3.5 & P6.md:
Conditions to automatically replace currently recommended adapter:
1. Score v1 is higher than current version by at least 2.0 points.
2. MAE is at least 1% better than both base reference model and naive flat persistence baseline.
3. No validation fold/segment has MAE worse than base by more than 5%.
4. 80% uncertainty interval coverage is between 65% and 95%, satisfying pinball criteria in composite loss.
5. Minimum 20 non-overlapping 24-hour prediction blocks evaluated.
6. 95% block bootstrap confidence interval for MAE improvement lies strictly on the positive side (CI_lower > 0).
7. Adapter saved, smoke-tested, and backed up successfully.

If any single condition fails, candidate is rejected, the current recommended adapter is preserved,
and detailed rejection reasons are saved to the decision audit log.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import logging
from pathlib import Path
import time
from typing import Any

from ..eval.types import FoldMetrics, ScoreReport
from ..model.manifest import AdapterManifest
from ..model.store import AdapterStore
from .bootstrap import BlockBootstrapResult, compute_block_bootstrap_ci

logger = logging.getLogger(__name__)

DEFAULT_AUDIT_REPORT_DIR = Path("var/paxg_lab/audit_reports")
DEFAULT_BACKUP_DIR = Path("var/paxg_lab/backups")


@dataclass
class GatekeeperDecision:
    """Decision audit outcome for candidate adapter verification."""

    candidate_id: str
    timeframe: str
    current_recommended_id: str | None
    accepted: bool
    verdict: str  # "ACCEPTED_NEW_RECOMMENDED" or "REJECTED_PRESERVE_CURRENT"
    check_score_improvement: bool
    check_mae_vs_base_and_naive: bool
    check_no_segment_worse_5pct: bool
    check_coverage_80_range: bool
    check_min_independent_blocks: bool
    check_bootstrap_ci_positive: bool
    check_smoke_test_and_backup: bool
    criteria_details: dict[str, Any] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    bootstrap_summary: dict[str, Any] = field(default_factory=dict)
    backup_zip_path: str | None = None
    audit_report_path: str | None = None
    evaluated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_json(self, output_path: str | Path) -> Path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        return p


class LoRAGatekeeper:
    """Evaluates candidate test verification results against winning criteria."""

    def __init__(
        self,
        store: AdapterStore,
        audit_dir: Path | str = DEFAULT_AUDIT_REPORT_DIR,
        backup_dir: Path | str = DEFAULT_BACKUP_DIR,
    ):
        self.store = store
        self.audit_dir = Path(audit_dir)
        self.backup_dir = Path(backup_dir)
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_candidate(
        self,
        candidate_manifest: AdapterManifest,
        candidate_test_report: ScoreReport,
        candidate_test_preds: Any,
        base_test_preds: Any,
        test_targets: Any,
        naive_test_mae: float,
        current_recommended_score: float = 0.0,
        perform_backup: bool = True,
        base_model: Any = None,
    ) -> GatekeeperDecision:
        """Evaluates whether candidate satisfies all 7 winner recognition criteria."""
        tf = candidate_manifest.timeframe
        cid = candidate_manifest.adapter_id
        current_rec_id = self.store.get_recommended(tf)

        # 1. Score improvement check: >= current_recommended + 2.0 points
        cand_score = float(candidate_test_report.score)
        score_diff = cand_score - current_recommended_score
        check_score = bool(score_diff >= 2.0)

        # 2. MAE vs Base & Naive: better by at least 1%
        cand_mae = float(candidate_test_report.overall_weighted_mae)
        base_mae_comp = candidate_test_report.baseline_comparisons.get("base_overall_weighted_mae")
        if base_mae_comp is None:
            # Look in test metrics
            if candidate_test_report.test_metrics:
                base_mae_comp = cand_mae / max(candidate_test_report.test_metrics.relative_mae_to_base, 1e-6)
            else:
                base_mae_comp = cand_mae

        base_mae = float(base_mae_comp)
        mae_vs_base_ratio = cand_mae / max(base_mae, 1e-6)
        mae_vs_naive_ratio = cand_mae / max(naive_test_mae, 1e-6)

        # Must be at least 1% better than both base and naive (ratio <= 0.99)
        check_mae = bool(mae_vs_base_ratio <= 0.99 and mae_vs_naive_ratio <= 0.99)

        # 3. No fold/segment worse than base by more than 5% (ratio <= 1.05)
        worst_fold_ratio = 0.0
        check_no_worse = True
        all_folds_to_check: list[FoldMetrics] = []
        if candidate_test_report.fold_metrics:
            all_folds_to_check.extend(candidate_test_report.fold_metrics)
        if candidate_test_report.test_metrics:
            all_folds_to_check.append(candidate_test_report.test_metrics)

        fold_ratios = {}
        for fm in all_folds_to_check:
            r = fm.relative_mae_to_base
            fold_ratios[str(fm.fold_id)] = r
            if r > worst_fold_ratio:
                worst_fold_ratio = r
            if r > 1.05:
                check_no_worse = False

        # 4. Coverage 80% range [0.65, 0.95]
        cov80 = float(candidate_test_report.coverage_80)
        check_cov80 = bool(0.65 <= cov80 <= 0.95)

        # 5. Independent blocks & Block Bootstrap CI
        bootstrap_res: BlockBootstrapResult | None = None
        check_blocks = False
        check_bootstrap = False
        bootstrap_dict = {}

        try:
            bootstrap_res = compute_block_bootstrap_ci(
                candidate_predictions=candidate_test_preds,
                baseline_predictions=base_test_preds,
                targets=test_targets,
                timeframe=tf,
                num_resamples=1000,
                seed=42,
                min_required_blocks=20,
            )
            check_blocks = bool(bootstrap_res.num_blocks >= 20)
            check_bootstrap = bool(bootstrap_res.ci_95_lower > 0.0)
            bootstrap_dict = bootstrap_res.to_dict()
        except Exception as exc:
            logger.warning("Block bootstrap evaluation failed or had insufficient blocks: %s", exc)
            bootstrap_dict = {"error": str(exc)}
            check_blocks = False
            check_bootstrap = False

        # Compile reasons
        reasons = []
        if not check_score:
            reasons.append(
                f"Score v1 improvement insufficient: candidate score {cand_score:.2f} vs current {current_recommended_score:.2f} "
                f"(diff {score_diff:+.2f} < +2.0 required)."
            )
        if not check_mae:
            reasons.append(
                f"MAE improvement insufficient (< 1% better): vs base={mae_vs_base_ratio:.3f} (req <= 0.99), "
                f"vs naive={mae_vs_naive_ratio:.3f} (req <= 0.99)."
            )
        if not check_no_worse:
            reasons.append(
                f"One or more validation segments worse than base by > 5%: max ratio={worst_fold_ratio:.3f} > 1.05."
            )
        if not check_cov80:
            reasons.append(
                f"Nominal 80% uncertainty coverage out of bounds: {cov80*100:.1f}% (required [65.0%, 95.0%])."
            )
        if not check_blocks:
            n_b = bootstrap_res.num_blocks if bootstrap_res else 0
            reasons.append(f"Insufficient independent 24h blocks: {n_b} < 20 required.")
        if not check_bootstrap:
            ci_l = bootstrap_res.ci_95_lower if bootstrap_res else float("-inf")
            reasons.append(f"95% block bootstrap CI lower bound not strictly positive: {ci_l:.4f} <= 0.")

        # 6. Check smoke test and backup
        check_smoke_backup = False
        backup_path_str: str | None = None

        all_numerical_passed = (
            check_score
            and check_mae
            and check_no_worse
            and check_cov80
            and check_blocks
            and check_bootstrap
        )

        if all_numerical_passed:
            try:
                cand_path = self.store.get_adapter_path(cid)
                if not (cand_path / "adapter_model.safetensors").exists():
                    raise FileNotFoundError(f"Missing adapter_model.safetensors in {cand_path}")
                if not (cand_path / "adapter_config.json").exists():
                    raise FileNotFoundError(f"Missing adapter_config.json in {cand_path}")

                if base_model is not None:
                    self.store.load_adapter(cid, base_model=base_model)

                if perform_backup:
                    backup_zip = self.backup_dir / f"recommended_{tf}_{cid}_{int(time.time())}.zip"
                    self.store.export_adapter_zip(cid, backup_zip)
                    backup_path_str = str(backup_zip)
                check_smoke_backup = True
            except Exception as exc:
                reasons.append(f"Smoke test load or backup export failed: {exc}")
                check_smoke_backup = False

        accepted = bool(all_numerical_passed and check_smoke_backup)
        verdict = "ACCEPTED_NEW_RECOMMENDED" if accepted else "REJECTED_PRESERVE_CURRENT"

        criteria_details = {
            "candidate_score": cand_score,
            "current_recommended_score": current_recommended_score,
            "score_diff": score_diff,
            "candidate_mae": cand_mae,
            "base_mae": base_mae,
            "naive_test_mae": naive_test_mae,
            "mae_vs_base_ratio": mae_vs_base_ratio,
            "mae_vs_naive_ratio": mae_vs_naive_ratio,
            "worst_segment_mae_ratio": worst_fold_ratio,
            "fold_ratios": fold_ratios,
            "coverage_80": cov80,
            "independent_24h_blocks": bootstrap_res.num_blocks if bootstrap_res else 0,
            "bootstrap_ci_95": [bootstrap_res.ci_95_lower, bootstrap_res.ci_95_upper] if bootstrap_res else None,
        }

        # Format audit report filename
        audit_file = self.audit_dir / f"audit_{tf}_{cid}_{int(time.time())}.json"

        decision = GatekeeperDecision(
            candidate_id=cid,
            timeframe=tf,
            current_recommended_id=current_rec_id,
            accepted=accepted,
            verdict=verdict,
            check_score_improvement=check_score,
            check_mae_vs_base_and_naive=check_mae,
            check_no_segment_worse_5pct=check_no_worse,
            check_coverage_80_range=check_cov80,
            check_min_independent_blocks=check_blocks,
            check_bootstrap_ci_positive=check_bootstrap,
            check_smoke_test_and_backup=check_smoke_backup,
            criteria_details=criteria_details,
            reasons=reasons,
            bootstrap_summary=bootstrap_dict,
            backup_zip_path=backup_path_str,
            audit_report_path=str(audit_file),
        )

        # If accepted, mark manifest verified and set recommended in store
        if accepted:
            logger.info("Candidate '%s' passed all criteria! Setting as RECOMMENDED adapter for %s.", cid, tf)
            candidate_manifest.is_verified = True
            cand_path = self.store.get_adapter_path(cid)
            candidate_manifest.save_json(cand_path / "paxg_manifest.json")
            self.store.set_recommended(cid, tf)
        else:
            logger.warning(
                "Candidate '%s' REJECTED. Preserving current recommended '%s' for %s. Reasons:\n%s",
                cid,
                current_rec_id,
                tf,
                "\n".join(f" - {r}" for r in reasons),
            )

        # Persist audit report
        decision.save_json(audit_file)
        return decision

"""Focused regressions for P6 adapter status, promotion, and persistence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import zipfile

import numpy as np
import torch
from safetensors.torch import save_file

from paxg_lab.data.snapshot import DatasetSnapshot, SnapshotMetadata
from paxg_lab.eval.types import FoldMetrics, ScoreReport
from paxg_lab.eval.predictor import TimesFM3Predictor
from paxg_lab.model.manifest import AdapterManifest
from paxg_lab.model.store import AdapterStore
from paxg_lab.model.train_spec import TrainSpec
from paxg_lab.queue.storage import GPUJobStorage
from paxg_lab.tune.bootstrap import BlockBootstrapResult
from paxg_lab.tune.gatekeeper import LoRAGatekeeper
from paxg_lab.tune.protocol import AutonomousTuningProtocol, SEEDS_MULTI_RUN


def _candidate(store: AdapterStore, adapter_id: str) -> tuple[AdapterManifest, Path]:
    path = store.get_adapter_path(adapter_id)
    path.mkdir(parents=True)
    manifest = AdapterManifest(
        adapter_id=adapter_id,
        timeframe="1h",
        horizon=24,
        context_len=256,
        feature_set="B",
        feature_columns=["close"],
        is_verified=False,
    )
    manifest.save_json(path / "paxg_manifest.json")
    save_file(
        {
            "base_model.model.seq_attn.0.query_proj.lora_A.weight": torch.zeros((4, 1280)),
            "base_model.model.seq_attn.0.query_proj.lora_B.weight": torch.zeros((1280, 4)),
        },
        path / "adapter_model.safetensors",
    )
    (path / "adapter_config.json").write_text(
        json.dumps({"r": 4, "target_modules": ["query_proj"], "peft_type": "LORA"}),
        encoding="utf-8",
    )
    return manifest, path


def _locked_report() -> SimpleNamespace:
    bootstrap = BlockBootstrapResult(
        timeframe="1h",
        block_size_candles=24,
        num_blocks=25,
        candidate_weighted_mae=9.8,
        baseline_weighted_mae=10.0,
        baseline_name="TimesFM3-Base",
        mean_improvement_usdt=0.2,
        relative_improvement_pct=2.0,
        ci_95_lower=0.1,
        ci_95_upper=0.3,
        is_significant_positive=True,
    )
    return SimpleNamespace(
        score_v1=8.5,
        current_recommended_score=0.0,
        score_diff=8.5,
        candidate_weighted_mae=9.8,
        base_weighted_mae=10.0,
        naive_weighted_mae=12.0,
        mae_vs_base_ratio=0.98,
        mae_vs_naive_ratio=9.8 / 12.0,
        worst_segment_ratio=1.02,
        segment_mae_ratios={"seg1": 0.98, "seg2": 1.02, "seg3": 1.0},
        coverage_80=0.82,
        bootstrap_result=bootstrap,
    )


def _snapshot() -> DatasetSnapshot:
    n = 3000
    timestamps = np.arange(1700000000000, 1700000000000 + n * 3600000, 3600000, dtype=np.int64)
    close = np.linspace(2000.0, 2200.0, n, dtype=np.float32)
    features_a = close[:, None]
    features_b = np.column_stack([features_a, np.ones((n, 8), dtype=np.float32)])
    features_c = np.column_stack([features_b, np.ones((n, 2), dtype=np.float32)])
    metadata = SnapshotMetadata(
        snapshot_id="adapter-audit",
        timeframe="1h",
        symbol="PAXGUSDT",
        start_time=int(timestamps[0]),
        end_time=int(timestamps[-1]),
        total_candles=n,
        feature_sets=["A", "B", "C"],
        created_at="2026-09-07T00:00:00Z",
        sha256="adapter_audit_snapshot",
    )
    return DatasetSnapshot(
        metadata=metadata,
        timestamps=timestamps,
        features_a=features_a,
        features_b=features_b,
        features_c=features_c,
    )


def test_save_adapter_preserves_unverified_status(tmp_path: Path):
    class FakePeftModel:
        def save_pretrained(self, output_dir: str) -> None:
            save_file({"weight": torch.zeros((1, 1))}, Path(output_dir) / "adapter_model.safetensors")
            (Path(output_dir) / "adapter_config.json").write_text("{}", encoding="utf-8")

    store = AdapterStore(tmp_path / "adapters", db_path=tmp_path / "registry.db")
    manifest = AdapterManifest(
        adapter_id="candidate",
        timeframe="1h",
        horizon=24,
        context_len=256,
        feature_set="B",
        feature_columns=["close"],
        is_verified=False,
    )
    path = store.save_adapter(FakePeftModel(), manifest, smoke_test=False)

    assert manifest.is_verified is False
    assert AdapterManifest.load_json(path / "paxg_manifest.json").is_verified is False


def test_gatekeeper_acceptance_produces_consistent_verified_zip(tmp_path: Path):
    store = AdapterStore(tmp_path / "adapters", db_path=tmp_path / "registry.db")
    manifest, path = _candidate(store, "accepted")
    gatekeeper = LoRAGatekeeper(store, audit_dir=tmp_path / "audit", backup_dir=tmp_path / "backups")

    decision = gatekeeper.evaluate_candidate(
        candidate_manifest=manifest,
        locked_report=_locked_report(),
        perform_backup=True,
    )

    assert decision.accepted is True
    assert store.get_recommended("1h") == "accepted"
    saved_manifest = AdapterManifest.load_json(path / "paxg_manifest.json")
    assert saved_manifest.is_verified is True
    for name, expected in saved_manifest.file_hashes.items():
        assert hashlib.sha256((path / name).read_bytes()).hexdigest() == expected

    with zipfile.ZipFile(decision.backup_zip_path) as archive:
        assert json.loads(archive.read("paxg_manifest.json"))['is_verified'] is True
        for name in ("paxg_manifest.json", "adapter_model.safetensors", "adapter_config.json", "checksums.sha256"):
            assert archive.read(name) == (path / name).read_bytes()
        checksums = archive.read("checksums.sha256").decode().splitlines()
        for line in checksums:
            expected, name = line.split(maxsplit=1)
            assert hashlib.sha256(archive.read(name)).hexdigest() == expected


def test_gatekeeper_backup_failure_rolls_back_unverified_candidate(tmp_path: Path):
    store = AdapterStore(tmp_path / "adapters", db_path=tmp_path / "registry.db")
    incumbent, incumbent_path = _candidate(store, "incumbent")
    incumbent_path.joinpath("paxg_manifest.json").write_text(
        json.dumps(incumbent.to_dict()), encoding="utf-8"
    )
    store.set_recommended("incumbent", "1h")
    candidate, candidate_path = _candidate(store, "rejected")
    gatekeeper = LoRAGatekeeper(store, audit_dir=tmp_path / "audit", backup_dir=tmp_path / "backups")

    def corrupt_export(_adapter_id: str, output_path: Path) -> Path:
        output_path.write_bytes(b"not a zip")
        return output_path

    with patch.object(store, "export_adapter_zip", side_effect=corrupt_export):
        decision = gatekeeper.evaluate_candidate(
            candidate_manifest=candidate,
            locked_report=_locked_report(),
            perform_backup=True,
        )

    assert decision.accepted is False
    assert store.get_recommended("1h") == "incumbent"
    assert AdapterManifest.load_json(candidate_path / "paxg_manifest.json").is_verified is False
    assert not list((tmp_path / "adapters").glob(".tmp_verify*"))
    assert not list((tmp_path / "backups").glob("*.zip"))


def test_gatekeeper_registry_failure_rolls_back_verified_stage(tmp_path: Path):
    store = AdapterStore(tmp_path / "adapters", db_path=tmp_path / "registry.db")
    _incumbent, incumbent_path = _candidate(store, "incumbent")
    store.set_recommended("incumbent", "1h")
    candidate, candidate_path = _candidate(store, "registry_failure")
    gatekeeper = LoRAGatekeeper(store, audit_dir=tmp_path / "audit", backup_dir=tmp_path / "backups")

    with patch.object(store, "set_recommended", side_effect=RuntimeError("db locked")):
        decision = gatekeeper.evaluate_candidate(
            candidate_manifest=candidate,
            locked_report=_locked_report(),
            perform_backup=True,
        )

    assert decision.accepted is False
    assert store.get_recommended("1h") == "incumbent"
    assert AdapterManifest.load_json(candidate_path / "paxg_manifest.json").is_verified is False
    assert not list((tmp_path / "adapters").glob(".tmp_verify*"))
    assert not list((tmp_path / "backups").glob("*.zip"))


def test_protocol_and_predictor_do_not_use_default_registry_db(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    default_db = Path("var/paxg_lab/paxg_lab.db")
    default_db.parent.mkdir(parents=True)
    with sqlite3.connect(default_db) as conn:
        conn.execute("CREATE TABLE sentinel (value TEXT NOT NULL)")
        conn.execute("INSERT INTO sentinel(value) VALUES ('untouched')")
    default_before = default_db.read_bytes()

    db_path = tmp_path / "protocol.db"
    protocol = AutonomousTuningProtocol(
        timeframe="1h",
        snapshot=_snapshot(),
        db_path=db_path,
        optuna_db_path=tmp_path / "optuna.db",
        adapter_store_dir=tmp_path / "adapters",
        audit_dir=tmp_path / "audit",
        backup_dir=tmp_path / "backups",
    )
    assert protocol.store.db_path == db_path
    candidate, candidate_path = _candidate(protocol.store, "protocol_candidate")
    decision = protocol.gatekeeper.evaluate_candidate(
        candidate_manifest=candidate,
        locked_report=_locked_report(),
        perform_backup=True,
    )
    assert decision.accepted is True
    assert protocol.store.get_recommended("1h") == "protocol_candidate"
    assert AdapterManifest.load_json(candidate_path / "paxg_manifest.json").is_verified is True
    assert default_db.read_bytes() == default_before

    adapter_dir = tmp_path / "file_only" / "adapter"
    adapter_dir.mkdir(parents=True)
    AdapterManifest(
        adapter_id="file_only",
        timeframe="1h",
        horizon=24,
        context_len=256,
        feature_set="B",
        feature_columns=["close"],
    ).save_json(adapter_dir / "paxg_manifest.json")
    predictor = object.__new__(TimesFM3Predictor)
    predictor.lora_model = None
    predictor.base_model = MagicMock()
    predictor.device = torch.device("cpu")
    with patch("paxg_lab.eval.predictor.AdapterStore") as store_cls:
        loaded_model = MagicMock()
        store_cls.return_value.load_adapter.return_value = (
            loaded_model,
            AdapterManifest.load_json(adapter_dir / "paxg_manifest.json"),
        )
        predictor.load_adapter(adapter_dir)
        store_cls.assert_called_once_with(base_dir=adapter_dir.parent, db_path=None)


def test_multi_seed_transition_persists_complete_audit(tmp_path: Path):
    snapshot = _snapshot()
    db_path = tmp_path / "multi_seed.db"
    protocol = AutonomousTuningProtocol(
        timeframe="1h",
        snapshot=snapshot,
        db_path=db_path,
        optuna_db_path=tmp_path / "optuna.db",
        adapter_store_dir=tmp_path / "adapters",
        audit_dir=tmp_path / "audit",
        backup_dir=tmp_path / "backups",
    )
    storage = GPUJobStorage(db_path)
    specs = [TrainSpec(timeframe="1h", lora_r=4), TrainSpec(timeframe="1h", lora_r=8)]
    storage.save_auto_tune_run(
        timeframe="1h",
        snapshot_path="mock",
        snapshot_hash=snapshot.metadata.sha256,
        phase="MULTI_SEED",
        top_specs_json=json.dumps([spec.to_dict() for spec in specs]),
        multi_seed_evaluations_json=json.dumps([]),
    )

    fold_metrics = [
        FoldMetrics(
            fold_id=fold,
            num_windows=10,
            weighted_mae=1.0,
            weighted_pinball=1.0,
            rmse=1.0,
            mae=1.0,
            coverage_80=0.8,
            mean_width_80=1.0,
            directional_accuracy=0.5,
            composite_loss=0.1 + fold * 0.01,
        )
        for fold in (1, 2, 3)
    ]
    report = ScoreReport(fold_metrics=fold_metrics)

    with patch.object(protocol, "get_or_compute_base_report", return_value=report):
        for _ in range(30):
            state = protocol.execute_step(custom_eval_fn=lambda _snapshot, _spec: report)
            if state["phase"] == "FINAL_FIT":
                break

    assert state["phase"] == "FINAL_FIT"
    fresh_state = GPUJobStorage(db_path).get_auto_tune_run("1h")
    persisted = json.loads(fresh_state["multi_seed_evaluations_json"])
    assert len(persisted) == len(specs)
    for entry in persisted:
        assert set(entry["seed_scores"]) == {str(seed) for seed in SEEDS_MULTI_RUN}
        assert set(entry["seed_best_epochs"]) == {str(seed) for seed in SEEDS_MULTI_RUN}
        assert entry["current_fold_losses"] == []
        assert entry["current_best_epochs"] == []
        assert entry["current_val_losses"] == []

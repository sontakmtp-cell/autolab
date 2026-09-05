"""Immutable Dataset Snapshot with SHA-256 cryptographic verification."""

from __future__ import annotations

import hashlib
import io
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

SNAPSHOTS_DIR = Path("var/paxg_lab/snapshots")


@dataclass
class SnapshotMetadata:
    snapshot_id: str
    timeframe: str  # "1h" or "4h"
    symbol: str
    start_time: int
    end_time: int
    total_candles: int
    feature_sets: list[str]
    created_at: str
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DatasetSnapshot:
    """Immutable, versioned dataset snapshot with SHA-256 integrity guarantee."""

    def __init__(
        self,
        metadata: SnapshotMetadata,
        timestamps: np.ndarray,
        features_a: np.ndarray,
        features_b: np.ndarray,
        features_c: np.ndarray | None = None,
    ):
        self.metadata = metadata
        self.timestamps = timestamps
        self.features_a = features_a
        self.features_b = features_b
        self.features_c = features_c

    @property
    def total_candles(self) -> int:
        return len(self.timestamps)

    @property
    def timeframe(self) -> str:
        return self.metadata.timeframe

    def get_features(self, feature_set: str = "B") -> np.ndarray:
        return self.get_feature_matrix(feature_set)

    def get_feature_matrix(self, feature_set: str = "B") -> np.ndarray:
        """Returns the feature matrix corresponding to feature set A, B, or C."""
        f_set = feature_set.upper().strip()
        if f_set == "A":
            return self.features_a
        elif f_set == "B":
            return self.features_b
        elif f_set == "C":
            if self.features_c is None:
                raise ValueError("Feature Set C is not included in this snapshot.")
            return self.features_c
        else:
            raise ValueError(f"Unknown feature set '{feature_set}'.")

    def to_dataframe(self, feature_set: str = "B") -> Any:
        """Converts snapshot feature matrix and timestamps into a pandas DataFrame."""
        import pandas as pd
        from .features import FEATURE_SPECS

        f_set = feature_set.upper().strip()
        matrix = self.get_feature_matrix(f_set)
        columns = list(FEATURE_SPECS[f_set].columns)
        df = pd.DataFrame(matrix, columns=columns)
        df.insert(0, "open_time", self.timestamps)
        return df

    @classmethod
    def create(
        cls,
        timeframe: str,
        timestamps: np.ndarray,
        features_a: np.ndarray,
        features_b: np.ndarray,
        features_c: np.ndarray | None = None,
        symbol: str = "PAXGUSDT",
    ) -> DatasetSnapshot:
        """Creates a new DatasetSnapshot and calculates its SHA-256 digest."""
        ts_arr = np.asarray(timestamps, dtype=np.int64)
        a_arr = np.asarray(features_a, dtype=np.float32)
        b_arr = np.asarray(features_b, dtype=np.float32)
        c_arr = np.asarray(features_c, dtype=np.float32) if features_c is not None else None

        f_sets = ["A", "B"]
        if c_arr is not None:
            f_sets.append("C")

        # Serialize to in-memory npz buffer to compute deterministic SHA-256
        buf = io.BytesIO()
        save_dict = {
            "timestamps": ts_arr,
            "features_a": a_arr,
            "features_b": b_arr,
        }
        if c_arr is not None:
            save_dict["features_c"] = c_arr

        np.savez_compressed(buf, **save_dict)
        npz_bytes = buf.getvalue()
        sha256_hash = hashlib.sha256(npz_bytes).hexdigest()

        snapshot_id = f"{symbol.lower()}_{timeframe}_{int(ts_arr[0])}_{int(ts_arr[-1])}_{sha256_hash[:8]}"
        created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        metadata = SnapshotMetadata(
            snapshot_id=snapshot_id,
            timeframe=timeframe,
            symbol=symbol,
            start_time=int(ts_arr[0]),
            end_time=int(ts_arr[-1]),
            total_candles=len(ts_arr),
            feature_sets=f_sets,
            created_at=created_at,
            sha256=sha256_hash,
        )

        return cls(
            metadata=metadata,
            timestamps=ts_arr,
            features_a=a_arr,
            features_b=b_arr,
            features_c=c_arr,
        )

    def save(self, base_dir: str | Path = SNAPSHOTS_DIR) -> Path:
        """Saves snapshot files atomically: data.npz and metadata.json."""
        out_dir = Path(base_dir) / self.metadata.snapshot_id
        out_dir.mkdir(parents=True, exist_ok=True)

        npz_path = out_dir / "data.npz"
        meta_path = out_dir / "metadata.json"

        save_dict = {
            "timestamps": self.timestamps,
            "features_a": self.features_a,
            "features_b": self.features_b,
        }
        if self.features_c is not None:
            save_dict["features_c"] = self.features_c

        np.savez_compressed(npz_path, **save_dict)

        # Verify hash of saved file
        with open(npz_path, "rb") as f:
            disk_hash = hashlib.sha256(f.read()).hexdigest()
        assert disk_hash == self.metadata.sha256, "Saved npz hash mismatch!"

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata.to_dict(), f, indent=2)

        return out_dir

    @classmethod
    def load(cls, snapshot_dir: str | Path, verify_hash: bool = True) -> DatasetSnapshot:
        """Loads snapshot from disk and verifies cryptographic integrity."""
        s_dir = Path(snapshot_dir)
        meta_path = s_dir / "metadata.json"
        npz_path = s_dir / "data.npz"

        if not meta_path.exists() or not npz_path.exists():
            raise FileNotFoundError(f"Snapshot directory incomplete at {s_dir}")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta_dict = json.load(f)
        metadata = SnapshotMetadata(**meta_dict)

        with open(npz_path, "rb") as f:
            npz_bytes = f.read()

        if verify_hash:
            actual_hash = hashlib.sha256(npz_bytes).hexdigest()
            if actual_hash != metadata.sha256:
                raise RuntimeError(
                    f"Snapshot integrity violation! Expected SHA-256 {metadata.sha256}, got {actual_hash}."
                )

        data = np.load(io.BytesIO(npz_bytes))
        timestamps = data["timestamps"]
        features_a = data["features_a"]
        features_b = data["features_b"]
        features_c = data["features_c"] if "features_c" in data else None

        return cls(
            metadata=metadata,
            timestamps=timestamps,
            features_a=features_a,
            features_b=features_b,
            features_c=features_c,
        )

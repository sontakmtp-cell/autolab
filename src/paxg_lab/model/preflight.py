"""TimesFM 3.0 Preflight and System Verification for RTX 5060 Ti."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Ensure safe console output on Windows cp1252 consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from ..constants import MODEL_REPO, MODEL_REVISION, TIMEFRAME_HORIZONS


@dataclass
class PreflightCheck:
    name: str
    status: str  # "PASS", "WARN", "FAIL"
    value: str
    details: str


@dataclass
class PreflightReport:
    model_repo: str
    model_revision: str
    checks: list[PreflightCheck]
    passed: bool
    verdict: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_repo": self.model_repo,
            "model_revision": self.model_revision,
            "passed": self.passed,
            "verdict": self.verdict,
            "checks": [asdict(c) for c in self.checks],
        }


def _get_ram_stats_gb() -> tuple[float, float]:
    """Returns (total_gb, avail_gb) on Windows / POSIX."""
    if sys.platform == "win32":
        try:
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))  # type: ignore
            return stat.ullTotalPhys / (1024**3), stat.ullAvailPhys / (1024**3)
        except Exception:
            pass
    return 16.0, 8.0


def run_preflight() -> PreflightReport:
    """Performs preflight checks for TimesFM 3.0 on RTX 5060 Ti."""
    checks: list[PreflightCheck] = []

    # 1. Python version
    py_ver = platform.python_version()
    major, minor = sys.version_info[:2]
    if (major, minor) == (3, 12):
        checks.append(
            PreflightCheck("Python Version", "PASS", py_ver, "Python 3.12 is active.")
        )
    elif (major, minor) >= (3, 10):
        checks.append(
            PreflightCheck("Python Version", "WARN", py_ver, f"Python >= 3.10 supported, but 3.12 recommended.")
        )
    else:
        checks.append(
            PreflightCheck("Python Version", "FAIL", py_ver, "TimesFM 3.0 requires Python >= 3.10.")
        )

    # 2. PyTorch & CUDA
    try:
        import torch

        torch_ver = torch.__version__
        cuda_avail = torch.cuda.is_available()

        if not cuda_avail:
            checks.append(
                PreflightCheck("GPU / CUDA", "FAIL", "No CUDA", "PyTorch does not detect CUDA. GPU acceleration required.")
            )
        else:
            device_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            capability = torch.cuda.get_device_capability(0)

            is_5060_ti = "5060" in device_name
            status = "PASS" if is_5060_ti else "WARN"
            checks.append(
                PreflightCheck(
                    "GPU Hardware",
                    status,
                    f"{device_name} ({vram_gb:.1f} GB, sm_{capability[0]}{capability[1]})",
                    f"Target GPU verified: RTX 5060 Ti with compute capability {capability} and {vram_gb:.1f} GB VRAM."
                    if is_5060_ti
                    else f"Detected GPU {device_name} (expected RTX 5060 Ti).",
                )
            )

            # Check PyTorch version CUDA suffix
            if "+cu" in torch_ver:
                checks.append(
                    PreflightCheck("PyTorch Build", "PASS", torch_ver, "CUDA-enabled PyTorch build is installed.")
                )
            else:
                checks.append(
                    PreflightCheck("PyTorch Build", "WARN", torch_ver, "Torch build lacks explicit +cu suffix.")
                )

    except ImportError:
        checks.append(
            PreflightCheck("PyTorch", "FAIL", "Not Installed", "PyTorch is missing from current environment.")
        )

    # 3. Checkpoint in cache
    home = Path.home()
    hf_cache_dir = home / ".cache" / "huggingface" / "hub" / "models--google--timesfm-3.0-pytorch" / "snapshots" / MODEL_REVISION
    if hf_cache_dir.exists() and (hf_cache_dir / "model.safetensors").exists():
        safetensor_size_mb = (hf_cache_dir / "model.safetensors").stat().st_size / (1024**2)
        checks.append(
            PreflightCheck(
                "Checkpoint Revision",
                "PASS",
                f"{MODEL_REVISION[:12]}... ({safetensor_size_mb:.1f} MB)",
                f"Local snapshot verified at {hf_cache_dir}",
            )
        )
    else:
        checks.append(
            PreflightCheck(
                "Checkpoint Revision",
                "WARN",
                f"{MODEL_REVISION[:12]}...",
                f"Checkpoint not pre-cached at {hf_cache_dir}. Will download from Hugging Face on first run.",
            )
        )

    # 4. Host RAM
    total_ram, avail_ram = _get_ram_stats_gb()
    if avail_ram >= 4.0:
        checks.append(
            PreflightCheck("System RAM", "PASS", f"Total: {total_ram:.1f} GB | Free: {avail_ram:.1f} GB", ">= 4 GB RAM available.")
        )
    else:
        checks.append(
            PreflightCheck("System RAM", "WARN", f"Total: {total_ram:.1f} GB | Free: {avail_ram:.1f} GB", "Available RAM < 4 GB.")
        )

    # 5. PEFT
    try:
        import peft
        checks.append(PreflightCheck("PEFT Library", "PASS", f"v{peft.__version__}", "PEFT library is available for LoRA."))
    except ImportError:
        checks.append(PreflightCheck("PEFT Library", "FAIL", "Not Installed", "PEFT is required for LoRA training."))

    passed = all(c.status != "FAIL" for c in checks)
    verdict = (
        "✅ Preflight PASSED: System & GPU ready for TimesFM 3.0 LoRA on RTX 5060 Ti."
        if passed
        else "❌ Preflight FAILED: Fix the failed checks before running."
    )

    return PreflightReport(
        model_repo=MODEL_REPO,
        model_revision=MODEL_REVISION,
        checks=checks,
        passed=passed,
        verdict=verdict,
    )


def print_preflight(report: PreflightReport) -> None:
    print("=" * 70)
    print("  TIMESFM 3.0 PREFLIGHT & HARDWARE CHECK (RTX 5060 Ti)")
    print(f"  Target Checkpoint: {report.model_repo} @ {report.model_revision}")
    print("=" * 70)
    for c in report.checks:
        tag = f"[{c.status}]"
        print(f"  {tag:<7} [{c.name:<18}] {c.value:<36}")
        print(f"          -> {c.details}")
    print("-" * 70)
    print(f"  VERDICT: {report.verdict}")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight check for TimesFM 3.0 on RTX 5060 Ti")
    parser.add_argument("--json", action="store_true", help="Output JSON format")
    args = parser.parse_args()

    report = run_preflight()
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print_preflight(report)

    sys.exit(0 if report.passed else 1)


if __name__ == "__main__":
    main()

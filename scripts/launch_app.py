"""One-step launch supervisor for PAXG Forecast Lab.

Starts the GPUScheduler background supervisor (if not already running)
and launches the Streamlit web dashboard bound locally to 127.0.0.1:8501.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import subprocess
import sys
import time

# Ensure repository root and src directory are on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from paxg_lab.queue.scheduler import GPUScheduler
from paxg_lab.queue.storage import GPUJobStorage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [Launcher] %(message)s",
)
logger = logging.getLogger("Launcher")


def main() -> int:
    parser = argparse.ArgumentParser(description="PAXG Forecast Lab One-step Launcher")
    parser.add_argument("--host", default="127.0.0.1", help="Host address to bind Streamlit server (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8501, help="Port to bind Streamlit server (default: 8501)")
    parser.add_argument("--db-path", default="var/paxg_lab/paxg_lab.db", help="Path to SQLite database")
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser automatically")
    args = parser.parse_args()

    db_path = Path(args.db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Check or start GPUScheduler background supervisor
    storage = GPUJobStorage(db_path)
    lease = storage.get_coordinator_lease()
    scheduler: GPUScheduler | None = None

    if lease is not None and (time.time() - float(lease.get("heartbeat", 0))) < 30.0:
        logger.info("Existing active GPUScheduler detected (PID %s). Reusing active coordinator.", lease.get("pid"))
    else:
        logger.info("Starting local GPUScheduler background supervisor thread...")
        try:
            scheduler = GPUScheduler(db_path=db_path, acquire_coordinator_lock=True)
            scheduler.start()
            logger.info("GPUScheduler started successfully.")
        except Exception as exc:
            logger.warning("Could not acquire coordinator lock (%s). Proceeding with UI launch.", exc)

    # 2. Build Streamlit command
    app_path = SRC_DIR / "paxg_lab" / "ui" / "app.py"
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        f"--server.address={args.host}",
        f"--server.port={args.port}",
        "--server.headless=true",
        "--server.fileWatcherType=none",
        "--browser.gatherUsageStats=false",
        "--theme.base=dark",
        "--theme.primaryColor=#f1c40f",
        "--theme.backgroundColor=#0e1117",
        "--theme.secondaryBackgroundColor=#161b22",
        "--theme.textColor=#e6edf3",
    ]

    logger.info("Launching Streamlit Web Dashboard: http://%s:%d", args.host, args.port)
    logger.info("Command: %s", " ".join(cmd))

    streamlit_proc: subprocess.Popen | None = None
    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{SRC_DIR};{REPO_ROOT}"
        streamlit_proc = subprocess.Popen(cmd, env=env)

        # Wait for process exit or interrupt
        ret = streamlit_proc.wait()
        return ret
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Shutting down gracefully...")
        if streamlit_proc:
            streamlit_proc.terminate()
            try:
                streamlit_proc.wait(timeout=5.0)
            except Exception:
                streamlit_proc.kill()
        return 0
    finally:
        if scheduler:
            logger.info("Stopping local GPUScheduler...")
            scheduler.stop()
            logger.info("GPUScheduler stopped.")


if __name__ == "__main__":
    sys.exit(main())

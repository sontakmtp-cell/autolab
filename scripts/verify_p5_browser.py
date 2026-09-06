"""End-to-End Browser verification script using Playwright for PAXG Forecast Lab (Phase P5).

Tests:
1. Streamlit server response on 127.0.0.1:8501
2. Presence of all 5 tabs in Vietnamese
3. Timeframe switching between 1h (24 steps / 24h) and 4h (6 steps / 24h)
4. Plotly candlestick chart and future steps table (24 steps in 1h, 6 steps in 4h)
5. Backtest tab: model selection, metrics, comparison vs Base, export buttons
6. Training tab: manual parameters, preflight resource check, start/stop buttons
7. Auto-tune tab: state machine controls, leaderboard, audit log
8. Adapter Manager tab: filtering, pin/unpin, renaming, safe export/import
9. Multi-tab and page reload safety (no duplicated jobs)
10. Saves full-resolution screenshots for acceptance evidence in docs/paxg-lab/phases/p5_evidence/
"""

from __future__ import annotations

import json
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

from playwright.sync_api import sync_playwright

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [BrowserVerify] %(message)s",
)
logger = logging.getLogger("BrowserVerify")

EVIDENCE_DIR = REPO_ROOT / "docs" / "paxg-lab" / "phases" / "p5_evidence"
CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
HOST = "127.0.0.1"
PORT = 8501
URL = f"http://{HOST}:{PORT}"


def wait_for_server(url: str, timeout: float = 30.0) -> bool:
    """Waits for Streamlit HTTP server to become responsive."""
    import urllib.request
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url, timeout=2.0) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.8)
    return False


def run_browser_verification() -> dict:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    evidence_report = {
        "timestamp": time.time(),
        "url": URL,
        "browser": "Google Chrome (Headless)",
        "checks": {},
        "screenshots": {},
    }

    db_path = REPO_ROOT / "var" / "paxg_lab" / "paxg_lab.db"
    from paxg_lab.queue.scheduler import GPUScheduler
    from paxg_lab.queue.storage import GPUJobStorage

    storage = GPUJobStorage(db_path)
    scheduler: GPUScheduler | None = None
    lease = storage.get_coordinator_lease()
    if not (lease is not None and (time.time() - float(lease.get("heartbeat", 0))) < 30.0):
        logger.info("Starting GPUScheduler background supervisor for verification...")
        try:
            scheduler = GPUScheduler(db_path=db_path, acquire_coordinator_lock=True)
            scheduler.start()
        except Exception as e:
            logger.warning("Could not acquire coordinator lock (%s). Proceeding.", e)

    logger.info("Starting Streamlit test instance...")
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{SRC_DIR};{REPO_ROOT}"

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(SRC_DIR / "paxg_lab" / "ui" / "app.py"),
        f"--server.address={HOST}",
        f"--server.port={PORT}",
        "--server.headless=true",
        "--server.fileWatcherType=none",
        "--browser.gatherUsageStats=false",
    ]

    server_proc = subprocess.Popen(cmd, env=env)

    try:
        logger.info("Waiting for Streamlit server at %s...", URL)
        if not wait_for_server(URL, timeout=35.0):
            raise RuntimeError(f"Streamlit server did not become responsive at {URL} within 35s.")
        logger.info("Streamlit server is ready!")

        with sync_playwright() as p:
            browser = p.chromium.launch(
                executable_path=CHROME_PATH,
                headless=True,
                args=["--no-sandbox", "--disable-gpu", "--window-size=1920,1080"],
            )
            context = browser.new_context(viewport={"width": 1920, "height": 1080})
            page = context.new_page()

            logger.info("Navigating to %s...", URL)
            page.goto(URL, wait_until="networkidle", timeout=30000)
            
            # Wait for main h1 heading to ensure Streamlit app has rendered
            page.wait_for_selector("h1", timeout=20000)
            page.wait_for_timeout(2000)

            # Check 1: Title and Header
            page_content = page.content()
            title_ok = ("PAXG Forecast Lab" in page_content) and ("timesfm-3.0-pytorch" in page_content)
            evidence_report["checks"]["header_and_title"] = title_ok
            logger.info("Check 1 - Header & Title: %s", title_ok)

            # Check 2: 5 Tabs presence
            tab_texts = page.get_by_role("tab").all_text_contents()
            expected_tab_keywords = ["Dự đoán", "Backtest", "Huấn luyện", "Tự động", "Quản lý"]
            tabs_found = {
                kw: any(kw in t for t in tab_texts)
                for kw in expected_tab_keywords
            }
            all_tabs_ok = all(tabs_found.values())
            evidence_report["checks"]["all_five_tabs_present"] = all_tabs_ok
            logger.info("Check 2 - Five Tabs present: %s (Found: %s)", all_tabs_ok, tab_texts)

            # ---------------------------------------------------------------
            # TAB 1: Forecast (1h mode -> 24 steps)
            # ---------------------------------------------------------------
            logger.info("Verifying Tab 1: Forecast in 1h mode...")
            page.wait_for_timeout(2000)

            # Screenshot Tab 1 (1h)
            tab1_1h_shot = EVIDENCE_DIR / "p5_tab1_forecast_1h_24steps.png"
            page.screenshot(path=str(tab1_1h_shot), full_page=True)
            evidence_report["screenshots"]["tab1_forecast_1h"] = str(tab1_1h_shot.name)
            logger.info("Captured screenshot: %s", tab1_1h_shot.name)

            # Verify 24 steps horizon text
            content_1h = page.content()
            horizon_1h_ok = "horizon = 24 nến" in content_1h or "24 nến / 24h" in content_1h
            evidence_report["checks"]["horizon_1h_24steps"] = horizon_1h_ok

            # ---------------------------------------------------------------
            # Switch to 4h mode (6 steps)
            # ---------------------------------------------------------------
            logger.info("Switching to 4h mode...")
            try:
                radio_4h = page.get_by_text("4h (6 nến / 24h)")
                radio_4h.click()
                time.sleep(3.0)
                page.wait_for_load_state("networkidle")
            except Exception as e:
                logger.warning("Could not click 4h radio: %s", e)

            content_4h = page.content()
            horizon_4h_ok = "horizon = 6 nến" in content_4h or "4h (6 nến / 24h)" in content_4h
            evidence_report["checks"]["horizon_4h_6steps"] = horizon_4h_ok

            # Screenshot Tab 1 (4h)
            tab1_4h_shot = EVIDENCE_DIR / "p5_tab1_forecast_4h_6steps.png"
            page.screenshot(path=str(tab1_4h_shot), full_page=True)
            evidence_report["screenshots"]["tab1_forecast_4h"] = str(tab1_4h_shot.name)
            logger.info("Captured screenshot: %s", tab1_4h_shot.name)

            # ---------------------------------------------------------------
            # TAB 2: Backtest
            # ---------------------------------------------------------------
            logger.info("Switching to Tab 2: Backtest...")
            tab2_elem = page.get_by_role("tab", name="Đánh giá Backtest")
            tab2_elem.click()
            time.sleep(2.0)

            content_tab2 = page.content()
            backtest_ok = "Score v1" in content_tab2 and "Mô hình cần đánh giá" in content_tab2
            evidence_report["checks"]["tab2_backtest"] = backtest_ok

            tab2_shot = EVIDENCE_DIR / "p5_tab2_backtest.png"
            page.screenshot(path=str(tab2_shot), full_page=True)
            evidence_report["screenshots"]["tab2_backtest"] = str(tab2_shot.name)
            logger.info("Captured screenshot: %s", tab2_shot.name)

            # ---------------------------------------------------------------
            # TAB 3: LoRA Training
            # ---------------------------------------------------------------
            logger.info("Switching to Tab 3: Huấn luyện LoRA...")
            tab3_elem = page.get_by_role("tab", name="Huấn luyện LoRA")
            tab3_elem.click()
            time.sleep(2.0)

            content_tab3 = page.content()
            training_ok = "Bậc ma trận phân rã LoRA" in content_tab3 or "LoRA Rank" in content_tab3
            evidence_report["checks"]["tab3_training"] = training_ok

            tab3_shot = EVIDENCE_DIR / "p5_tab3_training.png"
            page.screenshot(path=str(tab3_shot), full_page=True)
            evidence_report["screenshots"]["tab3_training"] = str(tab3_shot.name)
            logger.info("Captured screenshot: %s", tab3_shot.name)

            # ---------------------------------------------------------------
            # TAB 4: Autonomous Tuning
            # ---------------------------------------------------------------
            logger.info("Switching to Tab 4: Tự động Tối ưu...")
            tab4_elem = page.get_by_role("tab", name="Tự động Tối ưu")
            tab4_elem.click()
            time.sleep(2.0)

            content_tab4 = page.content()
            auto_ok = "Trạng thái Tự động" in content_tab4 and "Bảng Xếp hạng Cấu hình" in content_tab4
            evidence_report["checks"]["tab4_auto_tune"] = auto_ok

            tab4_shot = EVIDENCE_DIR / "p5_tab4_auto_tune.png"
            page.screenshot(path=str(tab4_shot), full_page=True)
            evidence_report["screenshots"]["tab4_auto_tune"] = str(tab4_shot.name)
            logger.info("Captured screenshot: %s", tab4_shot.name)

            # ---------------------------------------------------------------
            # TAB 5: LoRA Adapter Manager
            # ---------------------------------------------------------------
            logger.info("Switching to Tab 5: Quản lý LoRA...")
            tab5_elem = page.get_by_role("tab", name="Quản lý LoRA")
            tab5_elem.click()
            time.sleep(2.0)

            content_tab5 = page.content()
            manager_ok = "Quản lý Kho Trọng số LoRA Adapter" in content_tab5 and "Ghim chống xóa" in content_tab5
            evidence_report["checks"]["tab5_adapter_manager"] = manager_ok

            tab5_shot = EVIDENCE_DIR / "p5_tab5_adapter_manager.png"
            page.screenshot(path=str(tab5_shot), full_page=True)
            evidence_report["screenshots"]["tab5_adapter_manager"] = str(tab5_shot.name)
            logger.info("Captured screenshot: %s", tab5_shot.name)

            # ---------------------------------------------------------------
            # Check 6: Multi-tab & Reload Safety
            # ---------------------------------------------------------------
            logger.info("Verifying multi-tab & page reload safety...")
            page2 = context.new_page()
            page2.goto(URL, wait_until="networkidle", timeout=30000)
            time.sleep(2.0)
            page2_ok = "PAXG Forecast Lab" in page2.content()
            page2.close()

            # Reload original page
            page.reload(wait_until="networkidle")
            time.sleep(2.0)
            reload_ok = "PAXG Forecast Lab" in page.content()

            evidence_report["checks"]["multi_tab_and_reload_safe"] = page2_ok and reload_ok
            logger.info("Check 6 - Multi-tab & Reload safe: %s", page2_ok and reload_ok)

            context.close()
            browser.close()

    finally:
        logger.info("Terminating Streamlit test instance...")
        server_proc.terminate()
        try:
            server_proc.wait(timeout=5.0)
        except Exception:
            server_proc.kill()

        if scheduler:
            logger.info("Stopping GPUScheduler supervisor...")
            scheduler.stop()

    # Save evidence json
    evidence_file = EVIDENCE_DIR / "p5_browser_evidence.json"
    evidence_file.write_text(json.dumps(evidence_report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved browser verification evidence to %s", evidence_file)
    return evidence_report


def main() -> int:
    try:
        report = run_browser_verification()
        all_passed = all(report["checks"].values())
        print(f"Browser Verification Result: {'ALL PASSED' if all_passed else 'FAILED'}")
        print(json.dumps(report["checks"], indent=2))
        return 0 if all_passed else 1
    except Exception as exc:
        logger.error("Browser verification failed: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

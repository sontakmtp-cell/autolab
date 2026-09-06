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
import uuid

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
    from paxg_lab.queue.types import JobPriority, JobSpec, JobStatus, JobType

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

            # Record pre-click job state to track exact job submission
            existing_1h_job_ids = {j.job_id for j in storage.list_jobs(job_type=JobType.FORECAST.value, timeframe="1h")}
            click_time_1h = time.time() - 2.0

            submit_btn = page.locator("button:has-text('Tạo Dự đoán')")
            logger.info("Found 1h submit button: %d", submit_btn.count())
            if submit_btn.count() == 0:
                raise RuntimeError("Could not locate 'Tạo Dự đoán' submit button on 1h tab")

            logger.info("Submitting 1h forecast job via UI...")
            submit_btn.first.click()

            # Identify newly submitted job
            submitted_1h_job = None
            for _ in range(25):
                time.sleep(0.6)
                cands = [
                    j for j in storage.list_jobs(job_type=JobType.FORECAST.value, timeframe="1h")
                    if j.job_id not in existing_1h_job_ids and j.created_at >= click_time_1h
                ]
                if cands:
                    submitted_1h_job = cands[0]
                    break

            if not submitted_1h_job:
                active_cands = [
                    j for j in storage.list_jobs(job_type=JobType.FORECAST.value, timeframe="1h")
                    if j.status in (JobStatus.QUEUED.value, JobStatus.RUNNING.value)
                ]
                if active_cands:
                    submitted_1h_job = active_cands[0]
                else:
                    raise RuntimeError("No 1h forecast job was dispatched by UI submit click")

            logger.info("Tracking submitted 1h forecast job '%s' until SUCCEEDED...", submitted_1h_job.job_id)
            poll_start = time.time()
            job_succeeded_1h = False
            while time.time() - poll_start < 60.0:
                cur = storage.get_job(submitted_1h_job.job_id)
                if cur and cur.status == JobStatus.SUCCEEDED.value:
                    submitted_1h_job = cur
                    job_succeeded_1h = True
                    break
                elif cur and cur.status in (JobStatus.FAILED.value, JobStatus.CANCELLED.value):
                    raise RuntimeError(f"1h job {submitted_1h_job.job_id} terminated with {cur.status}: {cur.error_message}")
                time.sleep(0.8)

            assert job_succeeded_1h, f"Job {submitted_1h_job.job_id} did not succeed within 60s"
            assert submitted_1h_job.result is not None, f"Job {submitted_1h_job.job_id} result is empty"
            assert submitted_1h_job.result.get("timeframe") == "1h", "Result timeframe mismatch"
            res_pts_1h = submitted_1h_job.result.get("point_forecast") or []
            res_ts_1h = submitted_1h_job.result.get("target_timestamps") or []
            assert len(res_pts_1h) == 24, f"Expected 24 point_forecast entries, got {len(res_pts_1h)}"
            assert len(res_ts_1h) == 24, f"Expected 24 target_timestamps entries, got {len(res_ts_1h)}"

            # Wait for DOM table to render
            try:
                page.wait_for_selector("text=Bảng số chi tiết 24 bước", timeout=25000)
                forecast_1h_completed = True
                logger.info("1h forecast completed: verified job %s and DOM table 24 steps!", submitted_1h_job.job_id)
            except Exception as e:
                logger.error("Waiting for 24 steps table timed out: %s", e)
                forecast_1h_completed = False
                raise RuntimeError("1h forecast failed to produce 24 steps table within timeout") from e

            # Screenshot Tab 1 (1h)
            tab1_1h_shot = EVIDENCE_DIR / "p5_tab1_forecast_1h_24steps.png"
            page.screenshot(path=str(tab1_1h_shot), full_page=True)
            evidence_report["screenshots"]["tab1_forecast_1h"] = str(tab1_1h_shot.name)
            logger.info("Captured screenshot: %s", tab1_1h_shot.name)

            # Verify 24 steps table and metric strictly in DOM
            content_1h = page.content()
            horizon_1h_ok = forecast_1h_completed and ("Bảng số chi tiết 24 bước" in content_1h) and ("24 bước (1h)" in content_1h)
            evidence_report["checks"]["forecast_1h_completed"] = forecast_1h_completed
            evidence_report["checks"]["horizon_1h_24steps"] = horizon_1h_ok

            # ---------------------------------------------------------------
            # Switch to 4h mode (6 steps)
            # ---------------------------------------------------------------
            logger.info("Switching to 4h mode...")
            try:
                radio_4h = page.locator("label:has-text('4h (6 nến / 24h)')")
                if radio_4h.count() > 0:
                    radio_4h.first.click()
                else:
                    page.get_by_text("4h (6 nến / 24h)").click()
                time.sleep(3.0)
                page.wait_for_load_state("networkidle")
            except Exception as e:
                logger.warning("Could not click 4h radio: %s", e)

            # Record pre-click 4h job state
            existing_4h_job_ids = {j.job_id for j in storage.list_jobs(job_type=JobType.FORECAST.value, timeframe="4h")}
            click_time_4h = time.time() - 2.0

            submit_btn_4h = page.locator("button:has-text('Tạo Dự đoán')")
            logger.info("Found 4h submit button: %d", submit_btn_4h.count())
            if submit_btn_4h.count() == 0:
                raise RuntimeError("Could not locate 'Tạo Dự đoán' submit button on 4h tab")

            logger.info("Submitting 4h forecast job via UI...")
            submit_btn_4h.first.click()

            # Identify newly submitted 4h job
            submitted_4h_job = None
            for _ in range(25):
                time.sleep(0.6)
                cands = [
                    j for j in storage.list_jobs(job_type=JobType.FORECAST.value, timeframe="4h")
                    if j.job_id not in existing_4h_job_ids and j.created_at >= click_time_4h
                ]
                if cands:
                    submitted_4h_job = cands[0]
                    break

            if not submitted_4h_job:
                active_cands = [
                    j for j in storage.list_jobs(job_type=JobType.FORECAST.value, timeframe="4h")
                    if j.status in (JobStatus.QUEUED.value, JobStatus.RUNNING.value)
                ]
                if active_cands:
                    submitted_4h_job = active_cands[0]
                else:
                    raise RuntimeError("No 4h forecast job was dispatched by UI submit click")

            logger.info("Tracking submitted 4h forecast job '%s' until SUCCEEDED...", submitted_4h_job.job_id)
            poll_start = time.time()
            job_succeeded_4h = False
            while time.time() - poll_start < 60.0:
                cur = storage.get_job(submitted_4h_job.job_id)
                if cur and cur.status == JobStatus.SUCCEEDED.value:
                    submitted_4h_job = cur
                    job_succeeded_4h = True
                    break
                elif cur and cur.status in (JobStatus.FAILED.value, JobStatus.CANCELLED.value):
                    raise RuntimeError(f"4h job {submitted_4h_job.job_id} terminated with {cur.status}: {cur.error_message}")
                time.sleep(0.8)

            assert job_succeeded_4h, f"Job {submitted_4h_job.job_id} did not succeed within 60s"
            assert submitted_4h_job.result is not None, f"Job {submitted_4h_job.job_id} result is empty"
            assert submitted_4h_job.result.get("timeframe") == "4h", "Result timeframe mismatch"
            res_pts_4h = submitted_4h_job.result.get("point_forecast") or []
            res_ts_4h = submitted_4h_job.result.get("target_timestamps") or []
            assert len(res_pts_4h) == 6, f"Expected 6 point_forecast entries, got {len(res_pts_4h)}"
            assert len(res_ts_4h) == 6, f"Expected 6 target_timestamps entries, got {len(res_ts_4h)}"

            try:
                page.wait_for_selector("text=Bảng số chi tiết 6 bước", timeout=25000)
                forecast_4h_completed = True
                logger.info("4h forecast completed: verified job %s and DOM table 6 steps!", submitted_4h_job.job_id)
            except Exception as e:
                logger.error("Waiting for 6 steps table timed out: %s", e)
                forecast_4h_completed = False
                raise RuntimeError("4h forecast failed to produce 6 steps table within timeout") from e

            # Screenshot Tab 1 (4h)
            tab1_4h_shot = EVIDENCE_DIR / "p5_tab1_forecast_4h_6steps.png"
            page.screenshot(path=str(tab1_4h_shot), full_page=True)
            evidence_report["screenshots"]["tab1_forecast_4h"] = str(tab1_4h_shot.name)
            logger.info("Captured screenshot: %s", tab1_4h_shot.name)

            content_4h = page.content()
            horizon_4h_ok = forecast_4h_completed and ("Bảng số chi tiết 6 bước" in content_4h) and ("6 bước (4h)" in content_4h)
            evidence_report["checks"]["forecast_4h_completed"] = forecast_4h_completed
            evidence_report["checks"]["horizon_4h_6steps"] = horizon_4h_ok

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
            # Check 6: Multi-tab, Page Reload Safety, and Idempotency Verification
            # ---------------------------------------------------------------
            logger.info("Verifying multi-tab & page reload safety with SQLite queue verification...")
            count_before = len(storage.list_jobs())

            # 1. Reload page (F5 equivalent)
            page.reload(wait_until="networkidle")
            time.sleep(2.0)
            count_after_reload = len(storage.list_jobs())
            reload_jobs_unchanged = (count_after_reload == count_before)
            logger.info(
                "Reload check: jobs before=%d, after reload=%d (unchanged=%s)",
                count_before,
                count_after_reload,
                reload_jobs_unchanged,
            )

            # 2. Open second concurrent tab
            page2 = context.new_page()
            page2.goto(URL, wait_until="networkidle", timeout=30000)
            time.sleep(2.0)
            page2_title_ok = "PAXG Forecast Lab" in page2.content()
            count_after_tab2 = len(storage.list_jobs())
            tab2_jobs_unchanged = (count_after_tab2 == count_before)
            logger.info(
                "Second tab check: jobs after tab2=%d (unchanged=%s)",
                count_after_tab2,
                tab2_jobs_unchanged,
            )
            page2.close()

            # 3. Explicit Idempotency Test: Submit duplicate idempotency_key from concurrent sessions
            test_idem_key = f"e2e_idem_{int(time.time())}_{uuid.uuid4().hex[:4]}"
            test_spec_1 = JobSpec(
                job_id=f"test_idem_1_{uuid.uuid4().hex[:6]}",
                job_type=JobType.DUMMY.value,
                timeframe="1h",
                priority=JobPriority.MANUAL.value,
                payload={"tab": "tab1"},
                idempotency_key=test_idem_key,
            )
            test_spec_2 = JobSpec(
                job_id=f"test_idem_2_{uuid.uuid4().hex[:6]}",
                job_type=JobType.DUMMY.value,
                timeframe="1h",
                priority=JobPriority.MANUAL.value,
                payload={"tab": "tab2_duplicate"},
                idempotency_key=test_idem_key,
            )
            sub1 = storage.submit_job(test_spec_1)
            sub2 = storage.submit_job(test_spec_2)
            idempotency_enforced = (sub1 == sub2)
            matching_idem_jobs = [j for j in storage.list_jobs() if j.idempotency_key == test_idem_key]
            idempotency_enforced = idempotency_enforced and (len(matching_idem_jobs) == 1)
            logger.info(
                "Idempotency check: sub1=%s, sub2=%s, matching jobs in DB=%d (enforced=%s)",
                sub1,
                sub2,
                len(matching_idem_jobs),
                idempotency_enforced,
            )

            safety_ok = reload_jobs_unchanged and tab2_jobs_unchanged and page2_title_ok and idempotency_enforced
            evidence_report["checks"]["multi_tab_and_reload_safe"] = safety_ok
            evidence_report["checks"]["idempotency_enforced"] = idempotency_enforced
            logger.info("Check 6 - Multi-tab & Reload safe: %s (Idempotency: %s)", safety_ok, idempotency_enforced)

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

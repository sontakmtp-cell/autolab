"""Tab 4: Autonomous Tuning monitoring component with state machine controls, leaderboard, and validation logs."""

from __future__ import annotations

import json
from pathlib import Path
import time
import uuid

import pandas as pd
import streamlit as st

from ...constants import get_horizon_for_timeframe
from ...queue.storage import GPUJobStorage
from ...queue.types import AutoRunState, JobPriority, JobSpec, JobStatus, JobType
from ..state import (
    DEFAULT_DB_PATH,
    list_recent_jobs,
    timestamp_to_vietnam_str,
)

DEFAULT_OPTUNA_DB_PATH = Path("var/paxg_lab/optuna_studies.db")
DEFAULT_AUDIT_DIR = Path("var/paxg_lab/audit_reports")


def load_optuna_trials_for_timeframe(
    timeframe: str,
    optuna_db_path: Path = DEFAULT_OPTUNA_DB_PATH,
) -> list[dict]:
    """Loads trials directly from Optuna SQLite storage for the timeframe."""
    if not optuna_db_path.exists():
        return []
    try:
        import optuna
        storage_url = f"sqlite:///{optuna_db_path.resolve()}"
        study_summaries = optuna.get_all_study_summaries(storage=storage_url)
        target_summaries = [s for s in study_summaries if f"_{timeframe}_" in s.name or s.name.endswith(f"_{timeframe}")]
        if not target_summaries:
            return []

        trials_data = []
        for s in target_summaries:
            study = optuna.load_study(study_name=s.name, storage=storage_url)
            for t in study.trials:
                if t.state.name not in ("COMPLETE", "FAIL", "RUNNING"):
                    continue
                score = t.value if t.value is not None else float("nan")
                attrs = t.user_attrs
                trials_data.append({
                    "trial_number": t.number,
                    "study_name": s.name,
                    "score_v1": score,
                    "state": t.state.name,
                    "lora_r": attrs.get("lora_r", t.params.get("lora_r", 4)),
                    "learning_rate": attrs.get("learning_rate", t.params.get("learning_rate", 5e-5)),
                    "context_len": attrs.get("context_len", t.params.get("context_len", 256)),
                    "lora_dropout": attrs.get("lora_dropout", t.params.get("lora_dropout", 0.1)),
                    "feature_set": attrs.get("feature_set", t.params.get("feature_set", "B")),
                    "datetime_start": t.datetime_start.timestamp() if t.datetime_start else time.time(),
                })
        return sorted(trials_data, key=lambda x: (x["score_v1"] if not pd.isna(x["score_v1"]) else -999), reverse=True)
    except Exception:
        return []


def load_audit_reports_for_timeframe(timeframe: str, audit_dir: Path = DEFAULT_AUDIT_DIR) -> list[dict]:
    """Loads decision audit reports for candidate evaluation in the given timeframe."""
    if not audit_dir.exists():
        return []
    reports = []
    pattern = f"audit_{timeframe}_*.json"
    for p in sorted(audit_dir.glob(pattern), reverse=True):
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
                reports.append(data)
        except Exception:
            continue
    return reports


def render_auto_tune_tab(timeframe: str, db_path: Path = DEFAULT_DB_PATH) -> None:
    """Renders the Autonomous Tuning monitoring tab."""
    st.markdown("### 🤖 Giám sát Tự động Tối ưu Hóa (Optuna TPE)")
    st.caption("Giám sát trạng thái điều phối tự động, bảng xếp hạng các cấu hình siêu tham số và lý do công nhận/từ chối adapter.")

    storage = GPUJobStorage(db_path)
    current_state = storage.get_auto_run_state(timeframe)

    # 1. State Machine Banner & Controls
    col_status, col_btn_start, col_btn_stop = st.columns([2.2, 1.2, 1.2])

    state_colors = {
        AutoRunState.SEARCHING: "#2ecc71",
        AutoRunState.VALIDATING: "#3498db",
        AutoRunState.WAITING_DATA: "#f39c12",
        AutoRunState.WAITING_AUDIT: "#9b59b6",
        AutoRunState.PAUSED_ERROR: "#e74c3c",
        AutoRunState.STOPPED: "#95a5a6",
    }
    state_descriptions = {
        AutoRunState.SEARCHING: "Đang tìm kiếm cấu hình tốt bằng Optuna TPE (khám phá 10 lượt, tối đa 30 lượt)",
        AutoRunState.VALIDATING: "Đang kiểm chứng độc lập trên dữ liệu khóa kín (tối thiểu 20 khối 24h)",
        AutoRunState.WAITING_DATA: "Chờ dữ liệu nến mới từ sàn Binance (nghỉ GPU, khóa tập kiểm chứng cũ)",
        AutoRunState.WAITING_AUDIT: "Chờ xác nhận nghiệm thu",
        AutoRunState.PAUSED_ERROR: "Tạm dừng do lỗi tài nguyên (giữ ý định tự động)",
        AutoRunState.STOPPED: "Đã dừng tự động (Chờ lệnh người dùng)",
    }

    with col_status:
        color = state_colors.get(current_state, "#ffffff")
        desc = state_descriptions.get(current_state, "")
        st.markdown(
            f"""
            <div style="background-color: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 0.6rem 1rem;">
                <div style="font-size: 0.8rem; color: #8b949e;">Trạng thái Tự động ({timeframe}):</div>
                <div style="font-size: 1.2rem; font-weight: 700; color: {color};">
                    ● {current_state.value}
                </div>
                <div style="font-size: 0.82rem; color: #c9d1d9; margin-top: 0.2rem;">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_btn_start:
        st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)
        if st.button("▶ Bật Tự động", use_container_width=True, disabled=(current_state == AutoRunState.SEARCHING)):
            storage.set_auto_run_state(timeframe, AutoRunState.SEARCHING, allow_unstop=True)
            recent_auto = list_recent_jobs(job_type=JobType.AUTO_TRIAL.value, timeframe=timeframe, limit=5, db_path=db_path)
            has_pending = any(j.status in (JobStatus.QUEUED.value, JobStatus.RUNNING.value) for j in recent_auto)
            if not has_pending:
                snap_dir = Path("var/paxg_lab/snapshots")
                candidates = sorted(snap_dir.glob(f"paxgusdt_{timeframe}_*"))
                snap_path = str(candidates[-1]) if candidates else ""
                job_id = f"auto_tune_{timeframe}_{int(time.time())}"
                spec = JobSpec(
                    job_id=job_id,
                    job_type=JobType.AUTO_TRIAL.value,
                    timeframe=timeframe,
                    priority=JobPriority.AUTO.value,
                    payload={"timeframe": timeframe, "snapshot_path": snap_path, "max_trials": 30},
                    timeout_seconds=1200.0,
                )
                storage.submit_job(spec)
            st.success(f"Đã kích hoạt chế độ tự động cho khung {timeframe} và đưa công việc vào hàng đợi GPU!")
            st.rerun()

    with col_btn_stop:
        st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)
        if st.button("⏹ Dừng Tự động", use_container_width=True, disabled=(current_state == AutoRunState.STOPPED)):
            storage.set_auto_run_state(timeframe, AutoRunState.STOPPED)
            cancelled = storage.cancel_pending_auto_jobs(timeframe)
            running_job = storage.get_running_job()
            if running_job and running_job.priority == JobPriority.AUTO.value and (running_job.timeframe == timeframe or not running_job.timeframe):
                storage.request_cancel(running_job.job_id)
            st.warning(f"Đã dừng chế độ tự động khung {timeframe} (Hủy {cancelled} job chờ, yêu cầu dừng job đang chạy).")
            st.rerun()

    # Note regarding P6 implementation
    st.info(
        "💡 **Cơ chế Tối ưu Bayesian (PLAN 3.6 & P6):** "
        "Thuật toán Optuna TPE thực hiện 10 lượt khám phá đầu, tối đa 30 lượt mỗi đợt và dừng sớm nếu 12 lượt liên tiếp không cải thiện >= 0.5 điểm. "
        "Cấu hình đứng đầu được thẩm định qua 3 seed [42, 123, 2026], sau đó huấn luyện ứng viên cuối và kiểm chứng trên tối thiểu 20 khối độc lập dài 24 giờ. "
        "Chỉ khi vượt qua 7 tiêu chí Gatekeeper, adapter mới tự động thay thế bản khuyến nghị."
    )

    # 2. Leaderboard of Configurations
    st.markdown(f"#### 🏆 Bảng Xếp hạng Cấu hình Thử nghiệm ({timeframe})")

    optuna_trials = load_optuna_trials_for_timeframe(timeframe)
    leaderboard_data = []

    if optuna_trials:
        for idx, t in enumerate(optuna_trials):
            sc = t["score_v1"]
            sc_str = f"{sc:.2f}" if not pd.isna(sc) else "N/A"
            leaderboard_data.append({
                "Hạng": idx + 1,
                "Lượt thử": f"Trial #{t['trial_number']}",
                "Score v1": sc_str,
                "LoRA Rank": t["lora_r"],
                "Learning Rate": f"{t['learning_rate']:.1e}",
                "Context": t["context_len"],
                "Dropout": f"{t['lora_dropout']:.2f}",
                "Tập đặc trưng": t["feature_set"],
                "Trạng thái": t["state"],
                "Thời điểm": timestamp_to_vietnam_str(t["datetime_start"] * 1000),
            })
    else:
        # Fallback to DB queue auto jobs if Optuna study is running in queue
        auto_jobs = list_recent_jobs(job_type=JobType.AUTO_TRIAL.value, timeframe=timeframe, limit=20, db_path=db_path)
        if auto_jobs:
            for idx, j in enumerate(auto_jobs):
                res = j.result or {}
                sc = res.get("score")
                sc_str = f"{sc:.2f}" if sc is not None else "N/A"
                leaderboard_data.append({
                    "Hạng": idx + 1,
                    "Lượt thử": j.job_id,
                    "Score v1": sc_str,
                    "LoRA Rank": j.payload.get("train_spec", {}).get("lora_r", 4),
                    "Learning Rate": f"{j.payload.get('train_spec', {}).get('learning_rate', 5e-5):.1e}",
                    "Context": j.payload.get("train_spec", {}).get("context_len", 256),
                    "Dropout": f"{j.payload.get('train_spec', {}).get('lora_dropout', 0.1):.2f}",
                    "Tập đặc trưng": j.payload.get("train_spec", {}).get("feature_set", "B"),
                    "Trạng thái": j.status,
                    "Thời điểm": timestamp_to_vietnam_str(j.created_at * 1000),
                })

    if leaderboard_data:
        st.dataframe(pd.DataFrame(leaderboard_data), use_container_width=True, hide_index=True)
    else:
        st.info(f"Chưa có kết quả thử nghiệm tự động nào cho khung {timeframe}. Nhấn 'Bật Tự động' hoặc 'Gửi thử 1 Auto Job vào Hàng đợi' để khởi tạo.")

    # 3. Decision Audit Log (Lý do công nhận / từ chối)
    st.markdown("#### 📜 Nhật ký Thẩm định & Tiêu chí Thắng (PLAN 3.5 & P6)")
    audit_reports = load_audit_reports_for_timeframe(timeframe)
    audit_logs = []

    if audit_reports:
        for r in audit_reports:
            evaluated_at = r.get("evaluated_at", time.time())
            cid = r.get("candidate_id", "N/A")
            accepted = r.get("accepted", False)
            verdict_label = "✅ CÔNG NHẬN THẮNG" if accepted else "❌ TỪ CHỐI"
            crit = r.get("criteria_details", {})
            reasons = r.get("reasons", [])

            sc_cand = crit.get("candidate_score", 0.0)
            sc_diff = crit.get("score_diff", 0.0)
            cand_mae = crit.get("candidate_mae", 0.0)
            base_mae = crit.get("base_mae", 0.0)
            blocks = crit.get("independent_24h_blocks", 0)

            detail_parts = [
                f"Score: {sc_cand:.2f} (diff: {sc_diff:+.2f})",
                f"MAE: {cand_mae:.2f} USDT vs Base {base_mae:.2f} USDT",
                f"Khối 24h độc lập: {blocks} khối",
            ]
            if reasons:
                detail_parts.append(f"Lý do từ chối: {'; '.join(reasons)}")
            else:
                detail_parts.append("Đạt đầy đủ 7 tiêu chí thẩm định; đã cập nhật khuyến nghị và sao lưu.")

            audit_logs.append({
                "Thời điểm": timestamp_to_vietnam_str(evaluated_at * 1000),
                "Mã Ứng viên": cid,
                "Kết luận": verdict_label,
                "Chi tiết Đánh giá": " | ".join(detail_parts),
            })

    if audit_logs:
        st.dataframe(pd.DataFrame(audit_logs), use_container_width=True, hide_index=True)
    else:
        st.info("Chưa có nhật ký thẩm định nào. Nhật ký sẽ được ghi nhận tự động khi các thử nghiệm hoàn thành.")

    # 4. Dispatch Dry-Run Auto Trial (For test verification of queue interaction)
    with st.expander("🧪 Thử nghiệm Gửi Công việc Tự động (Dry-run test)", expanded=False):
        st.write("Gửi một công việc AUTO_TRIAL với mức ưu tiên thấp (Priority 3) để kiểm tra luân phiên và hàng đợi:")
        if st.button("Gửi thử 1 Auto Job vào Hàng đợi", key=f"btn_dry_auto_{timeframe}"):
            if current_state == AutoRunState.STOPPED:
                st.error("Không thể gửi việc tự động khi trạng thái đang là STOPPED. Vui lòng bấm 'Bật Tự động'.")
            else:
                dry_id = f"auto_test_{timeframe}_{uuid.uuid4().hex[:6]}"
                dummy_spec = JobSpec(
                    job_id=dry_id,
                    job_type=JobType.DUMMY.value,
                    timeframe=timeframe,
                    priority=JobPriority.AUTO.value,
                    payload={"steps": 3, "step_sleep": 0.2},
                )
                storage.submit_job(dummy_spec)
                st.success(f"Đã gửi công việc `{dry_id}` với mức ưu tiên AUTO vào hàng đợi!")
                st.rerun()

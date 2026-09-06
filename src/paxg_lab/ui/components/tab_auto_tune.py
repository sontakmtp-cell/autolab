"""Tab 4: Autonomous Tuning monitoring component with state machine controls, leaderboard, and validation logs."""

from __future__ import annotations

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
        AutoRunState.SEARCHING: "Đang tìm kiếm cấu hình tốt bằng Optuna TPE",
        AutoRunState.VALIDATING: "Đang kiểm chứng độc lập trên dữ liệu khóa kín",
        AutoRunState.WAITING_DATA: "Chờ dữ liệu nến mới từ sàn Binance (nghỉ GPU)",
        AutoRunState.WAITING_AUDIT: "Chờ xác nhận nghiệm thu",
        AutoRunState.PAUSED_ERROR: "Tạm dừng do lỗi (giữ ý định tự động)",
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
            storage.set_auto_run_state(timeframe, AutoRunState.SEARCHING)
            st.success(f"Đã kích hoạt chế độ tự động cho khung {timeframe}!")
            st.rerun()

    with col_btn_stop:
        st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)
        if st.button("⏹ Dừng Tự động", use_container_width=True, disabled=(current_state == AutoRunState.STOPPED)):
            storage.set_auto_run_state(timeframe, AutoRunState.STOPPED)
            cancelled = storage.cancel_pending_auto_jobs(timeframe)
            st.warning(f"Đã dừng chế độ tự động khung {timeframe} (Hủy {cancelled} job chờ).")
            st.rerun()

    # Note regarding P5 scope and P6 roadmap
    st.info(
        "💡 **Lộ trình triển khai (PLAN.md):** Ở Giai đoạn P5, thẻ cung cấp giao diện giám sát tiến trình, "
        "bảng xếp hạng và cơ chế Bật/Dừng hàng đợi. Chu trình tìm kiếm tự động đa seed và khóa kín hoàn chỉnh bằng Optuna TPE "
        "sẽ được kích hoạt chính thức trong **Giai đoạn P6**."
    )

    # 2. Leaderboard of Configurations
    st.markdown(f"#### 🏆 Bảng Xếp hạng Cấu hình Thử nghiệm ({timeframe})")

    auto_jobs = list_recent_jobs(job_type=JobType.AUTO_TRIAL.value, timeframe=timeframe, limit=20, db_path=db_path)
    leaderboard_data = []

    # Populate leaderboard from real auto trial jobs in DB
    if auto_jobs:
        for idx, j in enumerate(auto_jobs):
            res = j.result or {}
            leaderboard_data.append({
                "Hạng": idx + 1,
                "Trial ID": j.job_id,
                "LoRA Rank": j.payload.get("train_spec", {}).get("lora_r", 4),
                "Learning Rate": f"{j.payload.get('train_spec', {}).get('learning_rate', 5e-5):.0e}",
                "Context": j.payload.get("train_spec", {}).get("context_len", 256),
                "Val Loss": f"{res.get('best_val_loss', 0.0):.6f}" if "best_val_loss" in res else "N/A",
                "Trạng thái": j.status,
                "Thời gian tạo": timestamp_to_vietnam_str(j.created_at * 1000),
            })

    if leaderboard_data:
        st.dataframe(pd.DataFrame(leaderboard_data), use_container_width=True, hide_index=True)
    else:
        st.info(f"Chưa có kết quả thử nghiệm tự động nào cho khung {timeframe}. Nhấn 'Bật Tự động' hoặc 'Gửi thử 1 Auto Job vào Hàng đợi' để khởi tạo.")

    # 3. Decision Audit Log (Lý do công nhận / từ chối)
    st.markdown("#### 📜 Nhật ký Thẩm định & Tiêu chí Thắng (PLAN 3.5)")
    audit_logs = []
    if auto_jobs:
        for j in auto_jobs:
            res = j.result or {}
            time_str = timestamp_to_vietnam_str(j.finished_at * 1000) if j.finished_at else timestamp_to_vietnam_str(j.created_at * 1000)
            if j.status == JobStatus.SUCCEEDED.value:
                val_loss = res.get("best_val_loss")
                val_str = f" (Val loss: {val_loss:.6f})" if val_loss is not None else ""
                audit_logs.append({
                    "Thời điểm": time_str,
                    "Mã Thử nghiệm": j.job_id,
                    "Kết luận": "✅ HOÀN THÀNH",
                    "Lý do": f"Thử nghiệm huấn luyện tự động thành công{val_str}. Checkpoint hợp lệ đã được lưu trữ an toàn.",
                })
            elif j.status == JobStatus.FAILED.value:
                audit_logs.append({
                    "Thời điểm": time_str,
                    "Mã Thử nghiệm": j.job_id,
                    "Kết luận": "❌ THẤT BẠI",
                    "Lý do": j.error_message or "Tiến trình worker báo lỗi trong quá trình thực thi.",
                })
            elif j.status == JobStatus.CANCELLED.value:
                audit_logs.append({
                    "Thời điểm": time_str,
                    "Mã Thử nghiệm": j.job_id,
                    "Kết luận": "⏹ ĐÃ DỪNG",
                    "Lý do": "Thử nghiệm đã được dừng an toàn theo yêu cầu người dùng.",
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

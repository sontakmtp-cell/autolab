"""Tab 3: LoRA Training component with manual parameter controls, preflight checks, Start/Stop buttons, and live loss visualization."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any
import uuid

import psutil
import streamlit as st

from ...constants import get_horizon_for_timeframe
from ...model.train_spec import TrainSpec
from ...queue.storage import GPUJobStorage
from ...queue.types import JobPriority, JobSpec, JobStatus, JobType
from ..charts import build_loss_chart
from ..state import (
    DEFAULT_DB_PATH,
    get_latest_snapshot_path,
    list_recent_jobs,
    timestamp_to_vietnam_str,
)


def render_training_tab(timeframe: str, db_path: Path = DEFAULT_DB_PATH) -> None:
    """Renders the LoRA Training tab."""
    st.markdown("### 🛠️ Huấn luyện LoRA Thủ công (TimesFM 3.0)")
    st.caption("Khóa trọng số base TimesFM 3.0, huấn luyện adapter PEFT/LoRA trên GPU NVIDIA với cơ chế dừng sớm (Early Stopping) và checkpoint tự động.")

    horizon = get_horizon_for_timeframe(timeframe)
    storage = GPUJobStorage(db_path)
    snapshot_path = get_latest_snapshot_path(timeframe)

    # 1. Resource & Snapshot Preflight Status
    with st.expander("🔍 Kiểm tra Tài nguyên & Dữ liệu Huấn luyện (Preflight)", expanded=True):
        col_res1, col_res2, col_res3 = st.columns(3)
        with col_res1:
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_name = torch.cuda.get_device_name(0)
                    total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    st.success(f"GPU: **{gpu_name}** ({total_mem:.1f} GB)")
                else:
                    st.warning("CUDA không khả dụng (Chạy CPU)")
            except Exception:
                st.info("PyTorch GPU: Đang kiểm tra")

        with col_res2:
            ram = psutil.virtual_memory()
            ram_free_gb = ram.available / (1024**3)
            if ram_free_gb >= 4.0:
                st.success(f"RAM khả dụng: **{ram_free_gb:.1f} GB** (Đạt yêu cầu &ge;4GB)")
            else:
                st.warning(f"RAM khả dụng: {ram_free_gb:.1f} GB (Khuyến nghị &ge;4GB)")

        with col_res3:
            if snapshot_path and snapshot_path.exists():
                snap_name = snapshot_path.name
                st.success(f"Snapshot ({timeframe}): **{snap_name[:24]}...**")
            else:
                st.error(f"Thiếu snapshot {timeframe}! Vui lòng kiểm tra kho.")

    # 2. Form for Manual Hyperparameters (PLAN 3.2)
    with st.form(key=f"form_lora_train_{timeframe}"):
        st.markdown("#### ⚙️ Thiết lập Tham số Huấn luyện")

        row1_col1, row1_col2, row1_col3, row1_col4 = st.columns(4)
        with row1_col1:
            context_len = st.selectbox(
                "Độ dài ngữ cảnh (Context):",
                options=[128, 256, 512],
                index=1,
                help="Số lượng nến quá khứ cấp cho mô hình (mặc định 256).",
            )
        with row1_col2:
            st.text_input(
                "Độ dài dự đoán (Horizon):",
                value=f"{horizon} nến (Khóa cứng)",
                disabled=True,
                help=f"Theo PLAN 2.3: Khung {timeframe} khóa cứng horizon = {horizon} (nhìn trước 24 giờ).",
            )
        with row1_col3:
            feature_set = st.selectbox(
                "Bộ đặc trưng đầu vào:",
                options=["B — Giá & Khối lượng/Taker (Mặc định)", "A — Chỉ giá đóng cửa", "C — Hợp đồng Futures (Mark & Funding)"],
                index=0,
                help="Bộ đặc trưng huấn luyện theo PLAN 2.2.",
            )
            feat_code = "B" if "B" in feature_set else ("A" if "A" in feature_set else "C")
        with row1_col4:
            history_days = st.selectbox(
                "Phạm vi lịch sử học:",
                options=["365 ngày (Khuyến nghị)", "180 ngày", "Toàn bộ lịch sử khả dụng"],
                index=0,
                help="Phạm vi lịch sử nến dùng để huấn luyện theo PLAN 3.2 (180, 365 ngày hoặc toàn bộ).",
            )
            if "180" in history_days:
                mapped_history_days: int | str = 180
            elif "365" in history_days:
                mapped_history_days = 365
            else:
                mapped_history_days = "all"

        row2_col1, row2_col2, row2_col3, row2_col4 = st.columns(4)
        with row2_col1:
            lora_rank = st.selectbox(
                "LoRA Rank (r):",
                options=[2, 4, 8, 16],
                index=1,
                help="Bậc ma trận phân rã LoRA (mặc định 4).",
            )
        with row2_col2:
            lora_alpha = st.number_input(
                "LoRA Alpha:",
                min_value=1,
                max_value=64,
                value=lora_rank * 2,
                help="Hệ số co giãn LoRA (mặc định 2 × rank).",
            )
        with row2_col3:
            learning_rate = st.select_slider(
                "Tốc độ học (Learning Rate):",
                options=[1e-5, 3e-5, 5e-5, 1e-4, 2e-4, 3e-4],
                value=5e-5,
                format_func=lambda x: f"{x:.0e}",
                help="Learning rate trong khoảng [1e-5, 3e-4] theo PLAN 3.2.",
            )
        with row2_col4:
            max_epochs = st.slider("Số vòng học tối đa (Epochs):", min_value=1, max_value=10, value=5)

        # Advanced Settings (PLAN 3.2)
        with st.expander("🛠️ Tham số Nâng cao (Batch GPU, Tích lũy Gradient, Dừng sớm, Dropout, Clipping, Decay)", expanded=False):
            adv_c1, adv_c2, adv_c3, adv_c4 = st.columns(4)
            with adv_c1:
                batch_size = st.selectbox("Batch size GPU:", options=[1, 2, 4], index=1)
                lora_dropout = st.slider(
                    "LoRA Dropout:",
                    min_value=0.0,
                    max_value=0.20,
                    value=0.10,
                    step=0.01,
                    format="%.2f",
                    help="Tỷ lệ dropout LoRA trong khoảng [0.0, 0.20] theo PLAN 3.2.",
                )
            with adv_c2:
                grad_accum = st.selectbox("Tích lũy gradient:", options=[1, 2, 4, 8, 16], index=3)
                eff_batch = batch_size * grad_accum
                st.caption(f"Batch hiệu dụng: **{eff_batch}**")
                weight_decay = st.slider(
                    "Weight Decay:",
                    min_value=0.0,
                    max_value=0.10,
                    value=0.01,
                    step=0.01,
                    format="%.2f",
                    help="Hệ số suy giảm trọng số AdamW trong [0.0, 0.10] theo PLAN 3.2.",
                )
            with adv_c3:
                early_stop_patience = st.slider(
                    "Số vòng dừng sớm:",
                    min_value=1,
                    max_value=4,
                    value=2,
                    help="Số epoch không cải thiện trước khi dừng sớm (PLAN 3.2: 1–4).",
                )
                grad_clip_norm = st.slider(
                    "Gradient Clipping Norm:",
                    min_value=0.5,
                    max_value=2.0,
                    value=1.0,
                    step=0.1,
                    format="%.1f",
                    help="Ngưỡng cắt chuẩn gradient trong [0.5, 2.0] theo PLAN 3.2.",
                )
            with adv_c4:
                seed = st.number_input("Ngẫu nhiên (Seed):", min_value=1, max_value=999999, value=42)

        st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
        btn_col, _ = st.columns([1.5, 3.5])
        with btn_col:
            submit_train = st.form_submit_button("🚀 Bắt đầu Huấn luyện", use_container_width=True)

    # 3. Stop Button (Outside Form to allow immediate interrupt)
    running_job = storage.get_running_job()
    active_train_job_id = st.session_state.get(f"active_train_job_id_{timeframe}")

    if running_job and running_job.job_type in (JobType.TRAIN.value, JobType.AUTO_TRIAL.value):
        col_st1, col_st2 = st.columns([1.5, 3.5])
        with col_st1:
            if st.button("🛑 Dừng Huấn luyện (Graceful Stop)", type="secondary", use_container_width=True):
                storage.request_cancel(running_job.job_id)
                st.warning(f"Đã gửi yêu cầu dừng an toàn cho công việc `{running_job.job_id}`. Đang bảo toàn checkpoint...")

    # 4. Handle Training Submission
    if submit_train:
        if not snapshot_path or not snapshot_path.exists():
            st.error(f"Lỗi: Không tìm thấy snapshot cho khung {timeframe}.")
            return

        job_id = f"train_{timeframe}_r{lora_rank}_{uuid.uuid4().hex[:6]}"
        train_spec = TrainSpec(
            timeframe=timeframe,
            horizon=horizon,
            context_len=context_len,
            feature_set=feat_code,
            lora_r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            weight_decay=weight_decay,
            early_stopping_patience=early_stop_patience,
            grad_clip_norm=grad_clip_norm,
            history_days=mapped_history_days,
            seed=seed,
        )

        payload = {
            "timeframe": timeframe,
            "train_spec": train_spec.to_dict(),
            "snapshot_path": str(snapshot_path),
            "smoke_test": True,
        }

        job_spec = JobSpec(
            job_id=job_id,
            job_type=JobType.TRAIN.value,
            timeframe=timeframe,
            priority=JobPriority.MANUAL.value,
            payload=payload,
            timeout_seconds=1800.0,
        )

        submitted_id = storage.submit_job(job_spec)
        st.session_state[f"active_train_job_id_{timeframe}"] = submitted_id
        st.success(f"Đã gửi lệnh huấn luyện `{submitted_id}` vào hàng đợi GPU!")
        st.rerun()

    # 5. Monitor Live Progress
    if active_train_job_id:
        track_job = storage.get_job(active_train_job_id)
        if track_job and track_job.timeframe == timeframe:
            if track_job.status in (JobStatus.QUEUED.value, JobStatus.RUNNING.value):
                st.markdown("#### ⏳ Tiến trình Huấn luyện Thời gian Thực")
                p_pct = float(track_job.progress_pct) / 100.0
                p_msg = track_job.progress_message or "Đang chuẩn bị dữ liệu và mô hình"
                st.progress(min(max(p_pct, 0.0), 1.0), text=f"{p_msg} ({int(p_pct * 100)}%)")

                # Auto-refresh loop
                time.sleep(1.5)
                st.rerun()

            elif track_job.status == JobStatus.SUCCEEDED.value:
                res = track_job.result or {}
                st.success(
                    f"🎉 Huấn luyện thành công! Adapter ID: `{res.get('adapter_id', 'N/A')}` &bull; "
                    f"Best Epoch: {res.get('best_epoch', 1)} &bull; Best Val Loss: {res.get('best_val_loss', 0.0):.6f}"
                )
                st.session_state[f"last_trained_adapter_id_{timeframe}"] = res.get("adapter_id")
                st.session_state[f"last_training_result_{timeframe}"] = res
                st.session_state.pop(f"active_train_job_id_{timeframe}", None)

            elif track_job.status == JobStatus.CANCELLED.value:
                st.warning("Huấn luyện đã được dừng an toàn theo yêu cầu. Checkpoint hợp lệ đã được bảo toàn.")
                st.session_state.pop(f"active_train_job_id_{timeframe}", None)

            elif track_job.status == JobStatus.FAILED.value:
                st.error(f"Huấn luyện thất bại: {track_job.error_message}")
                st.session_state.pop(f"active_train_job_id_{timeframe}", None)
        else:
            st.session_state.pop(f"active_train_job_id_{timeframe}", None)

    # 6. Training Loss History Chart
    st.markdown("#### 📈 Biểu đồ Loss Huấn luyện Gần nhất")
    training_history: list[dict[str, Any]] = []
    last_res = st.session_state.get(f"last_training_result_{timeframe}")
    if (
        last_res
        and isinstance(last_res, dict)
        and last_res.get("history")
        and last_res.get("timeframe", timeframe) == timeframe
    ):
        training_history = last_res["history"]
    else:
        recent_succeeded = storage.list_jobs(
            job_type=JobType.TRAIN.value, timeframe=timeframe, status=JobStatus.SUCCEEDED.value, limit=1
        )
        if recent_succeeded and recent_succeeded[0].result and recent_succeeded[0].result.get("history"):
            training_history = recent_succeeded[0].result["history"]

    if training_history:
        epochs = [int(h.get("epoch", i + 1)) for i, h in enumerate(training_history)]
        train_losses = [float(h.get("train_loss", 0.0)) for h in training_history]
        val_losses = [float(h.get("val_loss", 0.0)) for h in training_history]
        fig_loss = build_loss_chart(
            train_losses=train_losses,
            val_losses=val_losses,
            epochs=epochs,
            title=f"Đường cong Loss Huấn luyện LoRA Thực tế ({timeframe})",
        )
        st.plotly_chart(fig_loss, use_container_width=True)
    else:
        st.info("Chưa có kết quả huấn luyện nào trong phiên hiện tại. Biểu đồ Loss sẽ hiển thị sau khi hoàn thành lượt huấn luyện thực tế.")

    # 7. Recent Training Jobs
    with st.expander("📜 Danh sách các đợt huấn luyện gần nhất", expanded=False):
        recent_trains = list_recent_jobs(job_type=JobType.TRAIN.value, timeframe=timeframe, limit=10, db_path=db_path)
        if recent_trains:
            train_rows = []
            for j in recent_trains:
                res = j.result or {}
                train_rows.append({
                    "Mã Job": j.job_id,
                    "Thời gian": timestamp_to_vietnam_str(j.created_at * 1000),
                    "Rank": j.payload.get("train_spec", {}).get("lora_r", 4),
                    "Trạng thái": j.status,
                    "Val Loss": f"{res.get('best_val_loss', 0.0):.6f}" if "best_val_loss" in res else "N/A",
                    "Thời lượng": f"{(j.finished_at - j.started_at):.1f}s" if (j.finished_at and j.started_at) else "N/A",
                })
            st.dataframe(pd.DataFrame(train_rows), use_container_width=True, hide_index=True)

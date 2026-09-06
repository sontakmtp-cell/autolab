"""Tab 2: Backtest component with out-of-sample evaluation, in-sample overlap checks, Score v1 metrics, and comparison against Base."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any
import uuid

import pandas as pd
import streamlit as st

from ...constants import get_horizon_for_timeframe
from ...model.store import AdapterStore
from ...queue.storage import GPUJobStorage
from ...queue.types import JobPriority, JobSpec, JobStatus, JobType
from ..charts import build_backtest_error_chart
from ..state import (
    DEFAULT_DB_PATH,
    get_latest_snapshot_path,
    list_recent_jobs,
    timestamp_to_vietnam_str,
)


def render_backtest_tab(timeframe: str, db_path: Path = DEFAULT_DB_PATH) -> None:
    """Renders the Backtest tab."""
    st.markdown("### 📊 Đánh giá Sai số Ngoài Mẫu (Backtest & Score v1)")
    st.caption("Kiểm chứng chất lượng dự đoán trên dữ liệu ngoài mẫu (out-of-sample), so sánh trực quan với TimesFM 3.0 Base và Chuẩn giữ nguyên giá.")

    horizon = get_horizon_for_timeframe(timeframe)
    adapter_store = AdapterStore(db_path=db_path)
    all_adapters = adapter_store.list_adapters()
    valid_adapters = [a for a in all_adapters if a.timeframe == timeframe]
    recommended_id = adapter_store.get_recommended(timeframe)

    model_options = ["TimesFM 3.0 Base (Chuẩn không adapter)"]
    adapter_manifest_map: dict[str, Any] = {"TimesFM 3.0 Base (Chuẩn không adapter)": None}

    for a in valid_adapters:
        alias = adapter_store.get_alias(a.adapter_id)
        star = " ⭐ [Khuyến nghị]" if a.adapter_id == recommended_id else ""
        label = f"LoRA: {alias}{star} (ID: {a.adapter_id[:16]}...)"
        model_options.append(label)
        adapter_manifest_map[label] = a

    # Form to configure and run backtest
    with st.form(key=f"form_backtest_{timeframe}"):
        col_m, col_f, col_b = st.columns([2.2, 1.4, 1.2])

        with col_m:
            selected_model_label = st.selectbox(
                "Mô hình cần đánh giá:",
                options=model_options,
                index=0,
                help="Chọn TimesFM 3.0 Base hoặc adapter LoRA để đánh giá hiệu năng trên dữ liệu quá khứ ngoài mẫu.",
            )
            selected_manifest = adapter_manifest_map[selected_model_label]
            selected_adapter_id = selected_manifest.adapter_id if selected_manifest else None

        with col_f:
            batch_size = st.selectbox("Batch GPU:", options=[8, 16, 32], index=1, help="Số lượng cửa sổ xử lý đồng thời trên GPU.")

        with col_b:
            include_test = st.checkbox(
                "Mở tập test khóa kín (90 ngày cuối)",
                value=False,
                help="CHÚ Ý: PLAN 3.3 yêu cầu tập test chỉ được mở khi nghiệm thu ứng viên cuối cùng!",
            )

        st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
        submit_backtest = st.form_submit_button("📊 Chạy Đánh giá Backtest", use_container_width=True)

    # In-sample overlap warning check
    if selected_manifest is not None:
        tr = selected_manifest.training_range
        t_start = tr.get("start_time_ms")
        t_end = tr.get("end_time_ms")
        if t_start and t_end:
            st.info(
                f"ℹ️ **Phạm vi dữ liệu học của adapter:** Từ {timestamp_to_vietnam_str(t_start)} "
                f"đến {timestamp_to_vietnam_str(t_end)}. Backtest ngoài mẫu đánh giá trên các giai đoạn sau thời điểm này."
            )

    storage = GPUJobStorage(db_path)
    snapshot_path = get_latest_snapshot_path(timeframe)

    if submit_backtest:
        if not snapshot_path or not snapshot_path.exists():
            st.error(f"Lỗi: Không tìm thấy snapshot cho khung {timeframe}.")
            return

        job_id = f"bt_{timeframe}_{uuid.uuid4().hex[:8]}"
        idempotency_key = f"bt_{timeframe}_{selected_adapter_id or 'base'}_{int(time.time() // 60)}"

        adapter_dir = str(adapter_store.get_adapter_path(selected_adapter_id)) if selected_adapter_id else None
        feature_set = selected_manifest.feature_set if selected_manifest else "A"
        model_name = adapter_store.get_alias(selected_adapter_id) if selected_adapter_id else "TimesFM3-Base"

        payload = {
            "timeframe": timeframe,
            "snapshot_path": str(snapshot_path),
            "adapter_path": adapter_dir,
            "model_type": "lora" if selected_adapter_id else "base",
            "model_name": model_name,
            "feature_set": feature_set,
            "context_len": selected_manifest.context_len if selected_manifest else 256,
            "batch_size": batch_size,
            "include_locked_test": include_test,
            "is_base_reference": (selected_adapter_id is None),
        }

        job_spec = JobSpec(
            job_id=job_id,
            job_type=JobType.BACKTEST.value,
            timeframe=timeframe,
            priority=JobPriority.MANUAL.value,
            payload=payload,
            idempotency_key=idempotency_key,
        )

        submitted_id = storage.submit_job(job_spec)
        st.session_state[f"active_backtest_job_id_{timeframe}"] = submitted_id

    # Polling & displaying backtest results
    active_job_id = st.session_state.get(f"active_backtest_job_id_{timeframe}")
    if active_job_id:
        active_job = storage.get_job(active_job_id)
        if active_job and active_job.timeframe == timeframe:
            if active_job.status in (JobStatus.QUEUED.value, JobStatus.RUNNING.value):
                prog_msg = active_job.progress_message or "Đang chuẩn bị dữ liệu"
                prog_pct = int(active_job.progress_pct)
                with st.spinner(f"Đang chạy Backtest qua hàng đợi GPU: {prog_msg} ({prog_pct}%)..."):
                    time.sleep(1.0)
                    st.rerun()

            elif active_job.status == JobStatus.SUCCEEDED.value and active_job.result:
                st.session_state[f"last_backtest_report_{timeframe}"] = active_job.result
                st.session_state.pop(f"active_backtest_job_id_{timeframe}", None)
                st.success(f"Backtest hoàn thành thành công! Mã công việc: `{active_job_id}`")

            elif active_job.status == JobStatus.FAILED.value:
                st.error(f"Backtest thất bại: {active_job.error_message}")
                st.session_state.pop(f"active_backtest_job_id_{timeframe}", None)
        else:
            st.session_state.pop(f"active_backtest_job_id_{timeframe}", None)

    # Display Backtest Report
    report_data = st.session_state.get(f"last_backtest_report_{timeframe}")
    if not report_data:
        # Load from recent DB history if available
        prev_bt = storage.list_jobs(job_type=JobType.BACKTEST.value, timeframe=timeframe, limit=1)
        if prev_bt and prev_bt[0].status == JobStatus.SUCCEEDED.value and prev_bt[0].result and prev_bt[0].timeframe == timeframe:
            report_data = prev_bt[0].result
            st.session_state[f"last_backtest_report_{timeframe}"] = report_data

    if report_data:
        _render_backtest_report_view(report_data, timeframe)
    else:
        st.info("Nhấn **Chạy Đánh giá Backtest** để bắt đầu quy trình đánh giá ngoài mẫu qua hàng đợi GPU.")

    # Recent Backtest Jobs History
    with st.expander("📜 Lịch sử các lần đánh giá Backtest gần nhất", expanded=False):
        recent_bt_jobs = list_recent_jobs(job_type=JobType.BACKTEST.value, timeframe=timeframe, limit=10, db_path=db_path)
        if recent_bt_jobs:
            history = []
            for j in recent_bt_jobs:
                res = j.result or {}
                score_val = res.get("score", res.get("score_v1"))
                score_str = f"{float(score_val):.2f}" if score_val is not None else "N/A"
                history.append({
                    "Mã Job": j.job_id,
                    "Thời gian tạo": timestamp_to_vietnam_str(j.created_at * 1000),
                    "Mô hình": j.payload.get("model_name", "Base"),
                    "Trạng thái": j.status,
                    "Score v1": score_str,
                    "Thời gian chạy": f"{(j.finished_at - j.started_at):.1f}s" if (j.finished_at and j.started_at) else "N/A",
                })
            st.dataframe(pd.DataFrame(history), use_container_width=True, hide_index=True)


def _render_backtest_report_view(report: dict[str, Any], timeframe: str) -> None:
    """Renders comprehensive backtest cards, charts, comparisons, and export buttons."""
    rep_tf = report.get("timeframe")
    if rep_tf and str(rep_tf).lower() != str(timeframe).lower():
        st.warning(f"Báo cáo backtest thuộc khung {rep_tf}, không khớp với khung {timeframe} đang chọn.")
        return

    model_name = report.get("model_name", "TimesFM3")
    score_v1 = float(report.get("score", report.get("score_v1", 0.0)))
    weighted_mae = float(report.get("overall_weighted_mae", report.get("weighted_mae", 0.0)))
    weighted_pinball = float(report.get("overall_weighted_pinball", report.get("weighted_pinball", 0.0)))
    cov_80 = float(report.get("coverage_80", 0.0)) * 100.0
    dir_acc = float(report.get("directional_accuracy", 0.0)) * 100.0

    st.markdown(f"#### 🏆 Kết quả Đánh giá: `{model_name}` (Khung {timeframe})")

    # Metrics row
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Score v1 (Chuẩn Base=0.0)", f"{score_v1:+.2f}", help="Score v1 theo công thức PLAN 3.4. Điểm dương: tốt hơn Base.")
    with c2:
        st.metric("MAE có trọng số", f"${weighted_mae:.2f}")
    with c3:
        st.metric("Pinball có trọng số", f"{weighted_pinball:.2f}")
    with c4:
        st.metric("Độ bao phủ dải 80%", f"{cov_80:.1f}%", help="Mục tiêu danh nghĩa là 80% (q10 đến q90).")
    with c5:
        st.metric("Độ đúng hướng (DA)", f"{dir_acc:.1f}%")

    # Step-by-step breakdown table & chart
    col_chart, col_tbl = st.columns([1.3, 1.0])

    step_mae_dict: dict[int, float] = {}

    # Extract step metrics: read report-level step_mae first, fallback to fold_metrics
    raw_step_mae = report.get("step_mae", {})
    if raw_step_mae and isinstance(raw_step_mae, dict):
        for k, v in raw_step_mae.items():
            try:
                step_mae_dict[int(k)] = float(v)
            except (ValueError, TypeError):
                pass
    elif report.get("fold_metrics"):
        fold_metrics = report.get("fold_metrics", [])
        for fm in fold_metrics:
            s_dict = fm.get("step_mae", {})
            for step_str, val in s_dict.items():
                try:
                    s_int = int(step_str)
                    step_mae_dict[s_int] = step_mae_dict.get(s_int, 0.0) + float(val) / max(1, len(fold_metrics))
                except (ValueError, TypeError):
                    pass

    # Relevant steps according to PLAN 4.4
    key_steps = [1, 6, 12, 24] if timeframe == "1h" else [1, 2, 3, 6]

    with col_chart:
        if step_mae_dict:
            fig_err = build_backtest_error_chart(
                step_errors={s: step_mae_dict[s] for s in key_steps if s in step_mae_dict},
                base_step_errors=None,
                timeframe=timeframe,
                title=f"Sai số MAE tại các mốc trọng yếu ({timeframe})",
            )
            st.plotly_chart(fig_err, use_container_width=True)

    with col_tbl:
        st.markdown("**Bảng sai số các mốc nhìn trước:**")
        step_rows = []
        for s in key_steps:
            mae_val = step_mae_dict.get(s, 0.0)
            step_rows.append({
                "Mốc dự đoán": f"Bước {s} (+{s * (4 if timeframe == '4h' else 1)}h)",
                "MAE (USDT)": f"${mae_val:.2f}",
            })
        st.dataframe(pd.DataFrame(step_rows), use_container_width=True, hide_index=True)

    # Three-way Comparison Table: Candidate vs Base vs Naive (No hardcoded fake values)
    st.markdown("#### ⚖️ Bảng đối chiếu hiệu năng với các chuẩn tham chiếu")
    base_comps = report.get("baseline_comparisons", {})
    naive_w_mae = base_comps.get("naive_flat_eval_weighted_mae")

    is_base = bool(
        report.get("metadata", {}).get("is_base_reference")
        or model_name == "TimesFM3-Base"
        or ("Base" in model_name and "LoRA" not in model_name)
    )

    if is_base:
        base_mae_str = f"${weighted_mae:.2f}"
        base_pinball_str = f"{weighted_pinball:.2f}"
        base_cov_str = f"{cov_80:.1f}%"
        base_da_str = f"{dir_acc:.1f}%"
        base_note = "Mô hình hiện tại (Chuẩn đối chiếu)"
    else:
        base_mae = base_comps.get("base_overall_weighted_mae")
        base_pin = base_comps.get("base_overall_weighted_pinball")
        base_c = base_comps.get("base_coverage_80")
        base_d = base_comps.get("base_directional_accuracy")

        base_mae_str = f"${base_mae:.2f}" if base_mae is not None else "N/A"
        base_pinball_str = f"{base_pin:.2f}" if base_pin is not None else "N/A"
        base_cov_str = f"{base_c * 100:.1f}%" if base_c is not None else "N/A"
        base_da_str = f"{base_d * 100:.1f}%" if base_d is not None else "N/A"
        base_note = "Chuẩn đối chiếu (Score=0.00)"

    comp_rows = [
        {
            "Mô hình": model_name,
            "Score v1": f"{score_v1:+.2f}",
            "MAE có trọng số": f"${weighted_mae:.2f}",
            "Pinball Loss": f"{weighted_pinball:.2f}",
            "Độ bao phủ 80%": f"{cov_80:.1f}%",
            "Đúng hướng": f"{dir_acc:.1f}%",
            "Ghi chú": "Mô hình đang đánh giá",
        },
        {
            "Mô hình": "TimesFM 3.0 Base (Chuẩn)",
            "Score v1": "0.00",
            "MAE có trọng số": base_mae_str,
            "Pinball Loss": base_pinball_str,
            "Độ bao phủ 80%": base_cov_str,
            "Đúng hướng": base_da_str,
            "Ghi chú": base_note,
        },
        {
            "Mô hình": "Chuẩn giữ nguyên giá (Naive)",
            "Score v1": "N/A",
            "MAE có trọng số": f"${naive_w_mae:.2f}" if naive_w_mae is not None else "N/A",
            "Pinball Loss": "N/A",
            "Độ bao phủ 80%": "N/A",
            "Đúng hướng": "N/A",
            "Ghi chú": "Dự đoán giá tương lai = giá hiện tại (đo đạc thực tế)",
        },
    ]
    st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

    # Export Buttons
    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        json_str = json.dumps(report, indent=2, ensure_ascii=False)
        st.download_button(
            label="💾 Tải Báo cáo JSON Đầy đủ",
            data=json_str,
            file_name=f"paxg_backtest_{timeframe}_{model_name}_{int(time.time())}.json",
            mime="application/json",
            use_container_width=True,
        )

    with col_exp2:
        df_export = pd.DataFrame(comp_rows)
        csv_str = df_export.to_csv(index=False)
        st.download_button(
            label="📄 Tải Bảng Đối chiếu CSV",
            data=csv_str,
            file_name=f"paxg_backtest_summary_{timeframe}_{model_name}_{int(time.time())}.csv",
            mime="text/csv",
            use_container_width=True,
        )

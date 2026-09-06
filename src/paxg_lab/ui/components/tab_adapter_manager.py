"""Tab 5: LoRA Adapter Management component with pinning, renaming, recommended selection, safe zip export/import, and trash protection."""

from __future__ import annotations

import io
from pathlib import Path
import tempfile
import time
import uuid

import pandas as pd
import streamlit as st

from ...constants import TIMEFRAME_1H, TIMEFRAME_4H
from ...model.store import AdapterStore
from ..state import DEFAULT_DB_PATH, timestamp_to_vietnam_str


def render_adapter_manager_tab(current_timeframe: str, db_path: Path = DEFAULT_DB_PATH) -> None:
    """Renders the LoRA Adapter Management tab."""
    st.markdown("### 🗄️ Quản lý Kho Trọng số LoRA Adapter")
    st.caption("Quản lý vòng đời adapter: xem chi tiết, đổi tên, ghim bảo vệ, chọn bản khuyến nghị, xuất/nhập an toàn và khôi phục.")

    store = AdapterStore(db_path=db_path)
    all_manifests = store.list_adapters()

    # 1. Top Filter & Statistics
    col_f, col_cnt1, col_cnt2, col_cnt3 = st.columns([1.5, 1.2, 1.2, 1.2])

    with col_f:
        filter_tf = st.selectbox(
            "Lọc theo khung thời gian:",
            options=["Tất cả khung giờ", "1h", "4h"],
            index=0 if current_timeframe not in ("1h", "4h") else (1 if current_timeframe == "1h" else 2),
        )
        selected_filter = None if "Tất cả" in filter_tf else filter_tf

    filtered_manifests = [
        m for m in all_manifests
        if (selected_filter is None or m.timeframe == selected_filter)
    ]

    rec_1h = store.get_recommended(TIMEFRAME_1H)
    rec_4h = store.get_recommended(TIMEFRAME_4H)

    with col_cnt1:
        st.metric("Tổng số adapter", f"{len(all_manifests)} bản")
    with col_cnt2:
        alias_1h = store.get_alias(rec_1h) if rec_1h else "Chưa đặt"
        st.metric("Khuyến nghị 1h", alias_1h[:16])
    with col_cnt3:
        alias_4h = store.get_alias(rec_4h) if rec_4h else "Chưa đặt"
        st.metric("Khuyến nghị 4h", alias_4h[:16])

    # 2. Main Adapter Table
    st.markdown("#### 📋 Danh sách Adapter trong Hệ thống")
    if filtered_manifests:
        table_rows = []
        for m in filtered_manifests:
            meta = store.get_registry_metadata(m.adapter_id)
            is_pinned = meta.get("is_pinned", False)
            is_rec = (m.adapter_id == rec_1h if m.timeframe == "1h" else m.adapter_id == rec_4h)
            alias = meta.get("alias", m.adapter_id)

            badges = []
            if is_rec:
                badges.append("⭐ Khuyến nghị")
            if is_pinned:
                badges.append("📌 Đã ghim")

            table_rows.append({
                "Tên hiển thị / Bí danh": alias,
                "Khung giờ": m.timeframe,
                "Horizon": f"{m.horizon} nến",
                "Context": m.context_len,
                "Bộ đặc trưng": m.feature_set,
                "Rank / Alpha": f"r={m.lora_config.get('r', 4)} / a={m.lora_config.get('lora_alpha', 8)}",
                "Val Loss": f"{m.best_val_loss:.6f}",
                "Trạng thái": " ".join(badges) if badges else "Bình thường",
                "ID Gốc": m.adapter_id,
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
    else:
        st.info("Không có adapter nào phù hợp với bộ lọc.")

    # 3. Individual Adapter Actions Panel
    st.markdown("#### 🛠️ Thao tác trên Adapter Được chọn")
    if filtered_manifests:
        adapter_choices = [
            f"{store.get_alias(m.adapter_id)} ({m.timeframe}) — {m.adapter_id[:16]}..."
            for m in filtered_manifests
        ]
        choice_idx = st.selectbox("Chọn adapter để thao tác:", options=range(len(adapter_choices)), format_func=lambda i: adapter_choices[i])
        active_manifest = filtered_manifests[choice_idx]
        active_id = active_manifest.adapter_id
        active_meta = store.get_registry_metadata(active_id)
        current_pinned = active_meta.get("is_pinned", False)
        is_active_rec = (active_id == rec_1h if active_manifest.timeframe == "1h" else active_id == rec_4h)

        c_act1, c_act2, c_act3, c_act4 = st.columns(4)

        with c_act1:
            # Pin / Unpin button
            pin_label = "🔓 Bỏ ghim" if current_pinned else "📌 Ghim chống xóa"
            if st.button(pin_label, use_container_width=True):
                store.set_pinned(active_id, not current_pinned)
                st.success(f"Đã cập nhật trạng thái ghim cho `{active_id}`!")
                st.rerun()

        with c_act2:
            # Set Recommended button
            rec_btn_label = "✅ Đang khuyến nghị" if is_active_rec else "⭐ Chọn làm Khuyến nghị"
            if st.button(rec_btn_label, use_container_width=True, disabled=is_active_rec):
                store.set_recommended(active_id, active_manifest.timeframe)
                st.success(f"Đã thiết lập `{active_id}` làm adapter khuyến nghị cho khung {active_manifest.timeframe}!")
                st.rerun()

        with c_act3:
            # Export zip button
            export_tmp = Path(tempfile.gettempdir()) / f"{active_id}.zip"
            try:
                store.export_adapter_zip(active_id, export_tmp)
                with open(export_tmp, "rb") as f:
                    zip_bytes = f.read()
                st.download_button(
                    label="📦 Xuất gói (.zip an toàn)",
                    data=zip_bytes,
                    file_name=f"{active_id}.zip",
                    mime="application/zip",
                    use_container_width=True,
                )
            except Exception as e:
                st.error(f"Lỗi chuẩn bị gói xuất: {e}")

        with c_act4:
            # Move to trash button with safety protection
            if st.button("🗑️ Chuyển vào Thùng rác", use_container_width=True):
                try:
                    store.delete_adapter(active_id, use_trash=True)
                    st.success(f"Đã chuyển adapter `{active_id}` vào thùng rác an toàn.")
                    st.rerun()
                except ValueError as ve:
                    st.error(str(ve))

        # Rename / Alias Form
        with st.form(key=f"form_rename_{active_id}"):
            st.markdown("**Đổi tên hiển thị (Bí danh):**")
            col_txt, col_rn_btn = st.columns([3, 1])
            with col_txt:
                new_alias = st.text_input("Bí danh mới:", value=store.get_alias(active_id), label_visibility="collapsed")
            with col_rn_btn:
                submit_rename = st.form_submit_button("Lưu tên mới", use_container_width=True)

            if submit_rename and new_alias.strip():
                store.set_alias(active_id, new_alias.strip())
                st.success(f"Đã cập nhật bí danh thành `{new_alias.strip()}`!")
                st.rerun()

    # 4. Safe Import Section
    st.markdown("#### 📥 Nhập Adapter An toàn (Safe Import)")
    with st.expander("Tải lên tệp zip adapter để nhập vào kho", expanded=False):
        st.write("Hệ thống chỉ chấp nhận tệp zip chứa trọng số `.safetensors`, `.json` và `checksums.sha256`. Tuyệt đối từ chối mã python hoặc pickle.")
        uploaded_file = st.file_uploader("Chọn tệp zip:", type=["zip"])
        if uploaded_file is not None:
            if st.button("Xác minh & Nhập Adapter", type="primary"):
                tmp_zip = Path(tempfile.gettempdir()) / f"upload_{uuid.uuid4().hex[:8]}.zip"
                with open(tmp_zip, "wb") as f:
                    f.write(uploaded_file.getvalue())

                try:
                    imported_id = store.import_adapter_zip(tmp_zip, overwrite=False)
                    st.success(f"Nhập thành công adapter `{imported_id}`! Đã xác minh 100% SHA-256.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Nhập thất bại: {exc}")
                finally:
                    if tmp_zip.exists():
                        tmp_zip.unlink(missing_ok=True)

    # 5. Trash & Restoration Section
    trash_items = store.list_trash_adapters()
    if trash_items:
        with st.expander(f"♻️ Thùng rác ({len(trash_items)} adapter)", expanded=False):
            st.write("Các adapter bị xóa tạm thời được lưu trong thư mục `.trash/` và có thể khôi phục:")
            for t_item in trash_items:
                col_tr_name, col_tr_btn = st.columns([3, 1])
                with col_tr_name:
                    st.text(t_item)
                with col_tr_btn:
                    orig_id = t_item.rsplit("_", 1)[0] if "_" in t_item else t_item
                    if st.button("Khôi phục", key=f"restore_{t_item}"):
                        if store.restore_adapter(orig_id):
                            st.success(f"Đã khôi phục adapter `{orig_id}`!")
                            st.rerun()
                        else:
                            st.error(f"Khôi phục thất bại (có thể adapter cùng tên đã tồn tại).")

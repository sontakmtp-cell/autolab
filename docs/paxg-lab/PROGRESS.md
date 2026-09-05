# PAXG Forecast Lab — Tiến độ triển khai dự án

Tài liệu này theo dõi tiến độ thực hiện 8 giai đoạn (P0 đến P7) của dự án **PAXG Forecast Lab** theo kế hoạch tại [docs/paxg-lab/PLAN.md](file:///d:/AI/timesfm_b/docs/paxg-lab/PLAN.md).

---

## 1. Bảng tổng quan các giai đoạn

| Giai đoạn | Mô tả | Trạng thái | Bằng chứng & Tài liệu |
|---|---|---|---|
| **P0** | Tài liệu, môi trường riêng và chứng minh LoRA 3.0 trên RTX 5060 Ti | **HOÀN THÀNH** | [docs/paxg-lab/phases/P0.md](file:///d:/AI/timesfm_b/docs/paxg-lab/phases/P0.md) |
| **P1** | Kho dữ liệu Binance PAXGUSDT Futures, đặc trưng A/B/C và phân chia không rò rỉ | Chưa bắt đầu | [docs/paxg-lab/phases/P1.md](file:///d:/AI/timesfm_b/docs/paxg-lab/phases/P1.md) |
| **P2** | Base TimesFM 3.0, dự đoán 24/6 bước, backtest và Score v1 | Chưa bắt đầu | [docs/paxg-lab/phases/P2.md](file:///d:/AI/timesfm_b/docs/paxg-lab/phases/P2.md) |
| **P3** | Huấn luyện thủ công LoRA, checkpoint tốt nhất và kho adapter | Chưa bắt đầu | [docs/paxg-lab/phases/P3.md](file:///d:/AI/timesfm_b/docs/paxg-lab/phases/P3.md) |
| **P4** | Hàng đợi GPU một tiến trình, dừng, heartbeat, phục hồi | Chưa bắt đầu | [docs/paxg-lab/phases/P4.md](file:///d:/AI/timesfm_b/docs/paxg-lab/phases/P4.md) |
| **P5** | Giao diện Streamlit tiếng Việt đủ 5 thẻ, biểu đồ và điều khiển | Chưa bắt đầu | [docs/paxg-lab/phases/P5.md](file:///d:/AI/timesfm_b/docs/paxg-lab/phases/P5.md) |
| **P6** | Tự động tối ưu Optuna TPE, kiểm chứng kín, công nhận LoRA thắng | Chưa bắt đầu | [docs/paxg-lab/phases/P6.md](file:///d:/AI/timesfm_b/docs/paxg-lab/phases/P6.md) |
| **P7** | Chạy dài (>6h), kiểm tra rò rỉ, xử lý sự cố, bàn giao hoàn chỉnh | Chưa bắt đầu | [docs/paxg-lab/phases/P7.md](file:///d:/AI/timesfm_b/docs/paxg-lab/phases/P7.md) |

---

## 2. Quyết định kỹ thuật đã chốt tại P0

1. **Quy ước Horizon bắt buộc:**
   - Khung 1h: `horizon = 24` (24 nến = 24 giờ).
   - Khung 4h: `horizon = 6` (6 nến = 24 giờ).
   - Định nghĩa tập trung tại `paxg_lab.constants`, không rải rác giá trị cố định trong mã dùng chung.
2. **Môi trường biệt lập & Phần cứng:**
   - Môi trường ảo `.venv-paxg` (Python 3.12.12).
   - PyTorch chính thức `2.12.1+cu130` hỗ trợ kiến trúc Blackwell `sm_120` của card đồ họa **NVIDIA GeForce RTX 5060 Ti** (16 GB VRAM).
   - Danh mục thư viện được khóa chặt tại `docs/paxg-lab/requirements-lock.txt`.
3. **Checkpoint mô hình:**
   - Ghim cố định revision `43046b85ec22d584a13f8098c2ed39c889e129c2` của `google/timesfm-3.0-pytorch` (1.32 GB).
4. **Kiến trúc đường tính toán dùng chung:**
   - Trong `src/timesfm3/model.py`, tách lõi tính toán thành `_decode_core()` không dùng decorator `@torch.no_grad()`.
   - Cung cấp `decode()` (chế độ suy luận, bảo lưu `@torch.no_grad()`) và `forward_decode()` (chế độ huấn luyện với autograd).
   - Đảm bảo tính toán detrending, CPM RevIN, stitching và horizon truncation là 100% đồng nhất giữa học và dự đoán.
5. **Cấu hình LoRA & Hàm Loss:**
   - Sử dụng Hugging Face PEFT (`peft==0.20.0`), mục tiêu `["query_proj", "value_proj"]` của `seq_attn` và `var_attn`.
   - Khóa toàn bộ 100% trọng số base (`requires_grad = False`), chỉ mở 819,200 tham số adapter (chiếm 0.247% tổng số tham số).
   - Hàm loss kết hợp MAE trung vị và pinball loss của 9 phân vị, chuẩn hóa theo giá đóng cửa cuối của cửa sổ context, tính toán dưới dạng `float32`.
6. **Kiểm chứng tài nguyên thực tế trên RTX 5060 Ti:**
   - Đỉnh VRAM khi huấn luyện ở context 512, batch 2, horizon 24 nến chỉ đạt ~2,711.8 MB (~2.65 GB), cách rất xa giới hạn an toàn 12 GB trên card 16 GB.
   - Toàn bộ 54/54 test (unit tests và P0 proof tests) đều đạt 100%.

---

## 3. Lệnh tiếp tục cho giai đoạn tiếp theo (P1)

Sau khi nghiệm thu P0, chuyển sang P1 bằng lệnh:

```text
/goal Đọc docs/paxg-lab/PLAN.md và PROGRESS.md, thực hiện P1: kho dữ liệu Binance PAXGUSDT Futures 1h/4h, đặc trưng A/B/C, chất lượng dữ liệu, snapshot và phân chia không rò rỉ. Kiểm tra horizon 1h=24, 4h=6 xuyên suốt nhãn và ranh giới dữ liệu. Chỉ hoàn thành P1.
```

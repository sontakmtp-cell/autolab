# PAXG Forecast Lab — Tiến độ triển khai dự án

Tài liệu này theo dõi tiến độ thực hiện 8 giai đoạn (P0 đến P7) của dự án **PAXG Forecast Lab** theo kế hoạch tại [docs/paxg-lab/PLAN.md](file:///d:/AI/timesfm_b/docs/paxg-lab/PLAN.md).

---

## 1. Bảng tổng quan các giai đoạn

| Giai đoạn | Mô tả | Trạng thái | Bằng chứng & Tài liệu |
|---|---|---|---|
| **P0** | Tài liệu, môi trường riêng và chứng minh LoRA 3.0 trên RTX 5060 Ti | **HOÀN THÀNH** | [docs/paxg-lab/phases/P0.md](file:///d:/AI/timesfm_b/docs/paxg-lab/phases/P0.md) |
| **P1** | Kho dữ liệu Binance PAXGUSDT Futures, đặc trưng A/B/C và phân chia không rò rỉ | **HOÀN THÀNH** | [docs/paxg-lab/phases/P1.md](file:///d:/AI/timesfm_b/docs/paxg-lab/phases/P1.md) |
| **P2** | Base TimesFM 3.0, dự đoán 24/6 bước, backtest và Score v1 | Chưa bắt đầu | [docs/paxg-lab/phases/P2.md](file:///d:/AI/timesfm_b/docs/paxg-lab/phases/P2.md) |
| **P3** | Huấn luyện thủ công LoRA, checkpoint tốt nhất và kho adapter | Chưa bắt đầu | [docs/paxg-lab/phases/P3.md](file:///d:/AI/timesfm_b/docs/paxg-lab/phases/P3.md) |
| **P4** | Hàng đợi GPU một tiến trình, dừng, heartbeat, phục hồi | Chưa bắt đầu | [docs/paxg-lab/phases/P4.md](file:///d:/AI/timesfm_b/docs/paxg-lab/phases/P4.md) |
| **P5** | Giao diện Streamlit tiếng Việt đủ 5 thẻ, biểu đồ và điều khiển | Chưa bắt đầu | [docs/paxg-lab/phases/P5.md](file:///d:/AI/timesfm_b/docs/paxg-lab/phases/P5.md) |
| **P6** | Tự động tối ưu Optuna TPE, kiểm chứng kín, công nhận LoRA thắng | Chưa bắt đầu | [docs/paxg-lab/phases/P6.md](file:///d:/AI/timesfm_b/docs/paxg-lab/phases/P6.md) |
| **P7** | Chạy dài (>6h), kiểm tra rò rỉ, xử lý sự cố, bàn giao hoàn chỉnh | Chưa bắt đầu | [docs/paxg-lab/phases/P7.md](file:///d:/AI/timesfm_b/docs/paxg-lab/phases/P7.md) |

---

## 2. Quyết định kỹ thuật đã chốt tại P0 và P1

### Tại P0:
1. **Quy ước Horizon bắt buộc:**
   - Khung 1h: `horizon = 24` (24 nến = 24 giờ).
   - Khung 4h: `horizon = 6` (6 nến = 24 giờ).
   - Định nghĩa tập trung tại `paxg_lab.constants`.
2. **Môi trường biệt lập & Phần cứng:**
   - Môi trường ảo `.venv-paxg` (Python 3.12.12).
   - PyTorch chính thức `2.12.1+cu130` hỗ trợ kiến trúc Blackwell `sm_120` của card đồ họa **NVIDIA GeForce RTX 5060 Ti** (16 GB VRAM).
   - Khóa phiên bản tại `docs/paxg-lab/requirements-lock.txt`.
3. **Đường tính toán dùng chung:**
   - Tách `_decode_core()` trong `src/timesfm3/model.py`, cung cấp `forward_decode()` có autograd cho LoRA và giữ `decode()` cho suy luận.
   - Chứng minh Base invariance, gradient hợp lệ, nạp/lưu adapter đạt `diff = 0.00e+00`.

### Tại P1:
1. **Kho dữ liệu Binance Futures & SQLite:**
   - Tải đầy đủ lịch sử từ ngày niêm yết (27/03/2025) đến nay: 12,651 nến 1h và 3,162 nến 4h.
   - Lưu trữ tại `var/paxg_lab/paxg_lab.db`.
   - Lọc bỏ nến đang chạy (unclosed) và nến đầu tiên không đủ thời lượng niêm yết.
2. **Chất lượng dữ liệu OHLC & Đối chiếu chéo:**
   - 0 lỗi giá, 0 nến âm, 0 mốc trùng lặp.
   - 0 khoảng thiếu thời gian (0 gaps) trên cả 1h và 4h.
   - Đối chiếu chéo 100% nến 4h với 4 nến 1h tương ứng đạt tỷ lệ khớp hoàn hảo 100.0% (3,162 / 3,162), 0 sai lệch.
3. **Ba bộ đặc trưng A/B/C:**
   - Bộ A (1 biến): `close`.
   - Bộ B (9 biến): Bộ A + `log1p(quote_volume)` + `log(high/low)` + `ret_oc` + `taker_buy_ratio` + cyclical sin/cos giờ/thứ.
   - Bộ C (11 biến): Bộ B + `mark_close_basis` + `realized_funding_rate`. Ghép funding theo thời điểm công bố (`funding_time <= open_time`), không rò rỉ tương lai.
4. **Snapshot bất biến:**
   - Đóng gói dữ liệu thành snapshot `.npz` kèm mã băm SHA-256:
     - 1h: `paxgusdt_1h_1743073200000_1788613200000_837c9ee8` (SHA-256: `837c9ee88b...`)
     - 4h: `paxgusdt_4h_1743076800000_1788595200000_c9e0adaf` (SHA-256: `c9e0adaf3f...`)
5. **Phân chia chuỗi thời gian (SplitSpec):**
   - 90 ngày cuối: Kiểm chứng khóa kín (Test).
   - 90 ngày trước đó: 3 đoạn đánh giá ngoài mẫu (Eval folds, 30 ngày/fold).
   - Trước mỗi đoạn đánh giá: Tập huấn luyện (Train), 14 ngày cuối cho dừng sớm (Early Stopping).
   - Khoảng chống chồng lấn nhãn (Purge buffer): 24 nến cho 1h, 6 nến cho 4h.
6. **Kiểm thử tự động:**
   - Toàn bộ 62/62 tests vượt qua 100% (`pytest tests/paxg_lab/ src/timesfm3/`).

---

## 3. Lệnh tiếp tục cho giai đoạn tiếp theo (P2)

Sau khi nghiệm thu P1, chuyển sang P2 bằng lệnh:

```text
/goal Đọc bộ tài liệu docs/paxg-lab và thực hiện P2: dự đoán base TimesFM 3.0, 24 nến cho 1h và 6 nến cho 4h, cùng nhìn trước 24 giờ. Xây backtest và Score v1, kiểm tra thời gian, phân vị, trọng số và chống nhìn trước. Lưu báo cáo base cho cả hai khung; chỉ hoàn thành P2.
```

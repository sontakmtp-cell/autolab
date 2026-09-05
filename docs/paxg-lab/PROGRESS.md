# PAXG Forecast Lab — Tiến độ triển khai dự án

Tài liệu này theo dõi tiến độ thực hiện 8 giai đoạn (P0 đến P7) của dự án **PAXG Forecast Lab** theo kế hoạch tại [docs/paxg-lab/PLAN.md](file:///d:/AI/timesfm_b/docs/paxg-lab/PLAN.md).

---

## 1. Bảng tổng quan các giai đoạn

| Giai đoạn | Mô tả | Trạng thái | Bằng chứng & Tài liệu |
|---|---|---|---|
| **P0** | Tài liệu, môi trường riêng và chứng minh LoRA 3.0 trên RTX 5060 Ti | **HOÀN THÀNH** | [docs/paxg-lab/phases/P0.md](file:///d:/AI/timesfm_b/docs/paxg-lab/phases/P0.md) |
| **P1** | Kho dữ liệu Binance PAXGUSDT Futures, đặc trưng A/B/C và phân chia không rò rỉ | **HOÀN THÀNH** | [docs/paxg-lab/phases/P1.md](file:///d:/AI/timesfm_b/docs/paxg-lab/phases/P1.md) |
| **P2** | Base TimesFM 3.0, dự đoán 24/6 bước, backtest và Score v1 | **HOÀN THÀNH** | [docs/paxg-lab/phases/P2.md](file:///d:/AI/timesfm_b/docs/paxg-lab/phases/P2.md) |
| **P3** | Huấn luyện thủ công LoRA, checkpoint tốt nhất và kho adapter | Chưa bắt đầu | [docs/paxg-lab/phases/P3.md](file:///d:/AI/timesfm_b/docs/paxg-lab/phases/P3.md) |
| **P4** | Hàng đợi GPU một tiến trình, dừng, heartbeat, phục hồi | Chưa bắt đầu | [docs/paxg-lab/phases/P4.md](file:///d:/AI/timesfm_b/docs/paxg-lab/phases/P4.md) |
| **P5** | Giao diện Streamlit tiếng Việt đủ 5 thẻ, biểu đồ và điều khiển | Chưa bắt đầu | [docs/paxg-lab/phases/P5.md](file:///d:/AI/timesfm_b/docs/paxg-lab/phases/P5.md) |
| **P6** | Tự động tối ưu Optuna TPE, kiểm chứng kín, công nhận LoRA thắng | Chưa bắt đầu | [docs/paxg-lab/phases/P6.md](file:///d:/AI/timesfm_b/docs/paxg-lab/phases/P6.md) |
| **P7** | Chạy dài (>6h), kiểm tra rò rỉ, xử lý sự cố, bàn giao hoàn chỉnh | Chưa bắt đầu | [docs/paxg-lab/phases/P7.md](file:///d:/AI/timesfm_b/docs/paxg-lab/phases/P7.md) |

---

## 2. Quyết định kỹ thuật đã chốt tại P0, P1 và P2

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
   - Bộ C (11 biến): Bộ B + `mark_close_basis` + `realized_funding_rate`. Ghép funding theo thời điểm công bố (`funding_time <= open_time`), không rò rỉ tương lai. Nếu thiếu mark price, báo lỗi rõ ràng và không tự bù bằng trade close.
4. **Snapshot bất biến & Báo cáo chất lượng:**
   - Đóng gói dữ liệu thành snapshot `.npz` kèm mã băm SHA-256:
     - 1h: `paxgusdt_1h_1743073200000_1788613200000_837c9ee8` (SHA-256: `837c9ee88b...`)
     - 4h: `paxgusdt_4h_1743076800000_1788595200000_c9e0adaf` (SHA-256: `c9e0adaf3f...`)
   - Lưu trữ bền vững báo cáo kiểm tra chất lượng dữ liệu vào SQLite table `data_quality_reports`.
   - Hỗ trợ đồng bộ dữ liệu gia tăng (incremental sync) cho cả klines, mark klines và funding rates.
5. **Phân chia chuỗi thời gian (SplitSpec) & Trích xuất cửa sổ:**
   - `extract_windows()`: Tôn trọng tuyệt đối ranh giới `[start_idx, end_idx)`. Target đầu tiên thỏa mãn $origin + 1 \ge start\_idx$, toàn bộ horizon nằm gọn trong $[start\_idx, end\_idx)$. Bắt buộc truyền `timestamps` và khoảng cách interval, tự động loại bỏ bất kỳ cửa sổ nào chứa khoảng thiếu (gap) timestamp.
   - 90 ngày cuối: Kiểm chứng khóa kín (Test).
   - 90 ngày trước đó: 3 đoạn đánh giá ngoài mẫu (Eval folds, 30 ngày/fold).
   - Trước mỗi đoạn đánh giá: Tập huấn luyện (Train), 14 ngày cuối cho dừng sớm (Early Stopping).
   - Khoảng chống chồng lấn nhãn (Purge buffer): 24 nến cho 1h, 6 nến cho 4h.

### Tại P2:
1. **Dự đoán Base TimesFM 3.0 & Khớp mốc thời gian:**
   - 1h: dự đoán 24 nến tương lai (24 giờ); 4h: dự đoán 6 nến tương lai (24 giờ).
   - Mốc thời gian tương lai khớp chính xác thời gian mở nến thực tế `open_time`.
2. **Tính đơn điệu của phân vị & Khoảng bất định:**
   - Tự động sắp xếp 9 phân vị đảm bảo tính đơn điệu $q_{10} \le q_{20} \le \dots \le q_{90}$.
   - Độ bao phủ dải bất định danh nghĩa 80% $[q_{10}, q_{90}]$ đạt mức xuất sắc: **75.9%** (1h) và **79.7%** (4h).
3. **Công thức tính điểm Score v1:**
   - Thiết lập trọng số suy giảm thời gian: 0-6h (50%), 6-12h (30%), 12-24h (20%).
   - TimesFM 3.0 Base đạt chuẩn đối chiếu **Score v1 = 0.00**.
   - Phạt thêm đoạn đánh giá tệ nhất: $Score = 100 \times [1 - (0.80 \times \bar{L} + 0.20 \times L_{worst})]$.
4. **Động cơ Backtest (`BacktestEngine`):**
   - Xử lý theo lô (`batch_size=32/16`) trên GPU NVIDIA GeForce RTX 5060 Ti, hoàn thành 2,091 cửa sổ 1h trong 38.1s và 525 cửa sổ 4h trong 10.6s.
   - Đánh giá trên 3 fold ngoài mẫu độc lập và tập kiểm chứng khóa kín.
   - Lưu bằng chứng thực nghiệm đầy đủ tại `docs/paxg-lab/phases/p2_base_benchmark.json`.
5. **Kiểm thử tự động:**
   - Toàn bộ 75/75 tests vượt qua 100% (`pytest tests/paxg_lab/ src/timesfm3/`).

---

## 3. Lệnh tiếp tục cho giai đoạn tiếp theo (P3)

Sau khi nghiệm thu P2, chuyển sang P3 bằng lệnh:

```text
/goal Đọc bộ tài liệu docs/paxg-lab và thực hiện P3: huấn luyện thủ công LoRA cho TimesFM 3.0 trên RTX 5060 Ti, 24 nến cho 1h và 6 nến cho 4h, cùng nhìn trước 24 giờ. Thêm lưu checkpoint tốt nhất theo validation, kho adapter và manifest. Kiểm tra chống nhìn trước, không sửa base và tái lập được; chỉ hoàn thành P3.
```

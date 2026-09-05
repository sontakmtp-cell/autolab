# PAXG Forecast Lab — Tiến độ triển khai dự án

Tài liệu này theo dõi tiến độ thực hiện 8 giai đoạn (P0 đến P7) của dự án **PAXG Forecast Lab** theo kế hoạch tại [docs/paxg-lab/PLAN.md](file:///d:/AI/timesfm_b/docs/paxg-lab/PLAN.md).

---

## 1. Bảng tổng quan các giai đoạn

| Giai đoạn | Mô tả | Trạng thái | Bằng chứng & Tài liệu |
|---|---|---|---|
| **P0** | Tài liệu, môi trường riêng và chứng minh LoRA 3.0 trên RTX 5060 Ti | **HOÀN THÀNH** | [docs/paxg-lab/phases/P0.md](file:///d:/AI/timesfm_b/docs/paxg-lab/phases/P0.md) |
| **P1** | Kho dữ liệu Binance PAXGUSDT Futures, đặc trưng A/B/C và phân chia không rò rỉ | **HOÀN THÀNH** | [docs/paxg-lab/phases/P1.md](file:///d:/AI/timesfm_b/docs/paxg-lab/phases/P1.md) |
| **P2** | Base TimesFM 3.0, dự đoán 24/6 bước, backtest và Score v1 | **HOÀN THÀNH** | [docs/paxg-lab/phases/P2.md](file:///d:/AI/timesfm_b/docs/paxg-lab/phases/P2.md) |
| **P3** | Huấn luyện thủ công LoRA, checkpoint tốt nhất và kho adapter | **HOÀN THÀNH** | [docs/paxg-lab/phases/P3.md](file:///d:/AI/timesfm_b/docs/paxg-lab/phases/P3.md) |
| **P4** | Hàng đợi GPU một tiến trình, dừng, heartbeat, phục hồi | Chưa bắt đầu | [docs/paxg-lab/phases/P4.md](file:///d:/AI/timesfm_b/docs/paxg-lab/phases/P4.md) |
| **P5** | Giao diện Streamlit tiếng Việt đủ 5 thẻ, biểu đồ và điều khiển | Chưa bắt đầu | [docs/paxg-lab/phases/P5.md](file:///d:/AI/timesfm_b/docs/paxg-lab/phases/P5.md) |
| **P6** | Tự động tối ưu Optuna TPE, kiểm chứng kín, công nhận LoRA thắng | Chưa bắt đầu | [docs/paxg-lab/phases/P6.md](file:///d:/AI/timesfm_b/docs/paxg-lab/phases/P6.md) |
| **P7** | Chạy dài (>6h), kiểm tra rò rỉ, xử lý sự cố, bàn giao hoàn chỉnh | Chưa bắt đầu | [docs/paxg-lab/phases/P7.md](file:///d:/AI/timesfm_b/docs/paxg-lab/phases/P7.md) |

---

## 2. Quyết định kỹ thuật đã chốt tại P0, P1, P2 và P3

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
3. **Công thức tính điểm Score v1 & Kết nối Base End-to-End:**
   - Thiết lập trọng số suy giảm thời gian: 0-6h (50%), 6-12h (30%), 12-24h (20%).
   - Từng fold kết nối trực tiếp với `weighted_mae` và `weighted_pinball` của Base:
     $A_f = \text{MAE}_{cand} / \text{MAE}_{base}$, $Q_f = \text{Pinball}_{cand} / \text{Pinball}_{base}$, $L_f = 0.70 A_f + 0.30 Q_f$.
   - TimesFM 3.0 Base đạt chuẩn đối chiếu **Score v1 = 0.00**.
   - Phạt thêm đoạn đánh giá tệ nhất: $Score = 100 \times [1 - (0.80 \times \bar{L} + 0.20 \times L_{worst})]$.
   - Xử lý nến/fold thiếu thông tin (base error <= tick size): gắn cờ `insufficient_information=True`, loại khỏi phép chia và tổng hợp.
   - Fail-fast bắt buộc: Candidate bắt buộc phải cung cấp `base_reference_metrics`, nếu thiếu sẽ ném `ValueError` ngay lập tức, triệt tiêu nguy cơ âm thầm trả Score 0.
   - Khóa invariant Base: Chỉ chấp nhận Feature Set A, context_len 256, không có adapter loaded cho lượt chạy Base reference chuẩn.
4. **Động cơ Backtest (`BacktestEngine`) & Bảo vệ Tuyệt Đối Test Khóa Kín:**
   - Xử lý theo lô (`batch_size=32/16`) trên GPU NVIDIA GeForce RTX 5060 Ti.
   - Kịch bản benchmark chính thức `scripts/run_p2_benchmarks.py` và `run_full_backtest()` mặc định **chỉ chạy eval folds và không mở tập test khóa kín** (`include_locked_test=False`), loại trừ 100% rò rỉ tập test.
   - Tệp bằng chứng `docs/paxg-lab/phases/p2_base_benchmark.json` được tạo lại chỉ chứa dữ liệu các fold đánh giá, không chứa dữ liệu tập test.
   - Cung cấp API riêng biệt `run_locked_verification()` dành riêng cho ứng viên chiến thắng cuối cùng.
   - Kiểm tra tương thích chặt chẽ `ForecastRequest.adapter_path` với `TimesFM3Predictor.adapter_path`.
   - Bổ sung đầy đủ các breakdown phân tích theo PLAN: biến động < 2 tick, ngày thường vs cuối tuần, nhóm biến động thấp/cao, 87 khối dự báo độc lập 24h.

### Tại P3:
1. **Huấn luyện LoRA thủ công & Khóa Horizon:**
   - Lớp cấu hình `TrainSpec` xác thực và khóa cứng bắt buộc: `1h` $\rightarrow$ `horizon = 24`, `4h` $\rightarrow$ `horizon = 6`. Từ chối mọi cấu hình sai cặp hoặc ngoài dải cho phép.
   - Hàm loss kết hợp `combined_forecast_loss` trong `float32`: bằng trung bình sai số tuyệt đối của trung vị ($q_{50}$) cộng Pinball loss 9 phân vị, chuẩn hóa chia theo giá cuối ngữ cảnh $p_0$.
   - Vòng huấn luyện `LoRATrainer` hỗ trợ đầy đủ tham số: rank (4), alpha (8), dropout (0.10), lr (5e-5), batch size (2), gradient accumulation (8, effective=16), linear warmup và gradient clipping (1.0).
   - Cơ chế dừng sớm (early stopping) giám sát trên đoạn dừng sớm 14 ngày, tự động chụp snapshot trọng số tốt nhất (`best_checkpoint`) và khôi phục vào mô hình khi hoàn tất.
2. **Khắc phục triệt để bẫy Autograd Sqrt:**
   - Cố định hàm `update_running_stats` (`src/timesfm3/util.py`) và `linear_detrending` (`src/timesfm3/model.py`) bằng cách kẹp `torch.clamp_min(var, 1e-8)` trước các lệnh `torch.sqrt()`. Triệt tiêu hoàn toàn lỗi `SqrtBackward0 returned nan` do đạo hàm vô hạn tại 0.
3. **Kho Adapter nguyên tử & Kháng gián đoạn:**
   - `AdapterStore`: Lưu theo cơ chế nguyên tử 5 bước: `.tmp_{id}` $\rightarrow$ tính SHA-256 các file $\rightarrow$ ghi manifest $\rightarrow$ smoke test forward pass $\rightarrow$ đổi tên nguyên tử (`os.replace`) thành `{adapter_id}`.
   - Tiến trình bị ngắt đột ngột giữa lúc ghi không bao giờ làm hỏng các adapter cũ; cung cấp cơ chế `cleanup_stale_temp_dirs` dọn sạch thư mục rác an toàn.
4. **Manifest provenance & Kiểm tra tương thích nghiêm ngặt:**
   - `AdapterManifest` lưu cấu hình LoRA, phạm vi thời gian huấn luyện, snapshot hash, checkpoint tốt nhất, và bảng mã băm SHA-256 cho từng tệp.
   - Từ chối nạp nếu sai mã băm (tamper detection), sai horizon, sai khung thời gian, sai ngữ cảnh hoặc sai revision base.
   - `check_in_sample_overlap`: Nhận diện và phát ra cảnh báo rõ ràng `IN-SAMPLE OVERLAP DETECTED` kèm số giờ chồng lấn nếu phạm vi đánh giá giao cắt với tập dữ liệu huấn luyện.
5. **Bảo toàn Base Model Invariant:**
   - Kiểm thử `test_base_restoration_invariant`: Chuỗi chuyển đổi $\text{Base} \rightarrow \text{Adapter A} \rightarrow \text{Adapter B} \rightarrow \text{Base}$ cho đầu ra trùng khớp tuyệt đối ($|F_0 - F_0'| = 0.00e+00$), các tensor trọng số gốc khớp từng bit, sạch 100% các khóa LoRA.
6. **Bằng chứng thực nghiệm trên RTX 5060 Ti:**
   - Kịch bản `scripts/run_p3_training.py` đã huấn luyện thành công 2 adapter thực tế:
     - 1h (`horizon = 24`): `paxg_1h_r4_setB_seed42_20260905_164330` (`val_loss = 0.011693`, 89.9s).
     - 4h (`horizon = 6`): `paxg_4h_r4_setB_seed42_20260905_164447` (`val_loss = 0.012624`, 71.2s).
   - Toàn bộ bằng chứng được lưu tại `docs/paxg-lab/phases/p3_lora_proof.json`.
7. **Kiểm thử tự động:**
   - **90 tests collected: 90 passed** (`pytest tests/paxg_lab/ src/timesfm3/ -v`), 100% pass rate.

---

## 3. Lệnh tiếp tục cho giai đoạn tiếp theo (P4)

Sau khi nghiệm thu P3, chuyển sang P4 bằng lệnh:

```text
/goal Đọc bộ tài liệu docs/paxg-lab và thực hiện P4: xây dựng hàng đợi một tiến trình GPU, quản lý công việc (Queued, Running, Succeeded, Failed, Cancelled, Interrupted), cơ chế heartbeat phát hiện treo/tiến trình chết, dừng an toàn và phục hồi sau sự cố. Kiểm thử ngắt đột ngột, khóa luân phiên 1h/4h và tôn trọng quyền ưu tiên dự đoán; chỉ hoàn thành P4.
```

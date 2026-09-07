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
| **P4** | Hàng đợi GPU một tiến trình, dừng, heartbeat, phục hồi | **HOÀN THÀNH** | [docs/paxg-lab/phases/P4.md](file:///d:/AI/timesfm_b/docs/paxg-lab/phases/P4.md) |
| **P5** | Giao diện Streamlit tiếng Việt đủ 5 thẻ, biểu đồ và điều khiển | **HOÀN THÀNH** | [docs/paxg-lab/phases/P5.md](file:///d:/AI/timesfm_b/docs/paxg-lab/phases/P5.md) |
| **P6** | Tự động tối ưu Optuna TPE, kiểm chứng kín, công nhận LoRA thắng | **HOÀN THÀNH** | [docs/paxg-lab/phases/P6.md](file:///d:/AI/timesfm_b/docs/paxg-lab/phases/P6.md) |
| **P7** | Chạy dài (>6h), kiểm tra rò rỉ, xử lý sự cố, bàn giao hoàn chỉnh | Chưa bắt đầu | [docs/paxg-lab/phases/P7.md](file:///d:/AI/timesfm_b/docs/paxg-lab/phases/P7.md) |

---

## 2. Quyết định kỹ thuật đã chốt tại P0, P1, P2, P3, P4, P5 và P6

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
1. **Huấn luyện LoRA thủ công & Khóa Horizon chuẩn PLAN 3.2:**
   - Lớp cấu hình `TrainSpec` xác thực và khóa cứng bắt buộc: `1h` $\rightarrow$ `horizon = 24`, `4h` $\rightarrow$ `horizon = 6` (tự động suy luận theo `timeframe` nếu không truyền). Từ chối mọi cấu hình sai cặp hoặc ngoài dải quy định PLAN 3.2: `learning_rate` trong `[1e-5, 3e-4]`, `warmup_ratio` khóa chặt trong `[0.0, 0.10]` (tối đa 10% số bước cập nhật optimizer), `max_epochs` trong `[1, 10]`, `batch_size` trong `(1, 2, 4)`.
   - Hàm loss kết hợp `combined_forecast_loss` trong `float32`: bằng trung bình sai số tuyệt đối của trung vị ($q_{50}$) cộng Pinball loss 9 phân vị, chuẩn hóa chia theo giá cuối ngữ cảnh $p_0$.
   - Vòng huấn luyện `LoRATrainer` hỗ trợ đầy đủ tham số: rank (4), alpha (8), dropout (0.10), lr (5e-5), batch size (2), gradient accumulation (8, effective=16), linear warmup theo bước optimizer thực tế và gradient clipping (1.0).
   - Bảo vệ mô hình base sạch: `LoRATrainer` từ chối nhận base model đã chứa PEFT/LoRA.
   - Cơ chế dừng sớm (early stopping) giám sát trên đoạn dừng sớm 14 ngày, tự động chụp snapshot trọng số tốt nhất (`best_checkpoint`) và khôi phục vào mô hình khi hoàn tất.
2. **Khắc phục triệt để bẫy Autograd Sqrt:**
   - Cố định hàm `update_running_stats` (`src/timesfm3/util.py`) và `linear_detrending` (`src/timesfm3/model.py`) bằng cách kẹp `torch.clamp_min(var, 1e-8)` trước các lệnh `torch.sqrt()`. Triệt tiêu hoàn toàn lỗi `SqrtBackward0 returned nan` do đạo hàm vô hạn tại 0.
3. **Kho Adapter nguyên tử, Smoke Test & Kháng gián đoạn:**
   - `AdapterStore`: Lưu theo cơ chế nguyên tử 5 bước: `.tmp_{id}` $\rightarrow$ tính SHA-256 các file $\rightarrow$ ghi manifest $\rightarrow$ smoke test forward pass thật trên device (kiểm tra shape `(1, num_features, horizon, 9)` và tính hữu hạn) $\rightarrow$ ghi `checksums.sha256` sidecar $\rightarrow$ đổi tên nguyên tử (`os.replace`) thành `{adapter_id}`.
   - Tiến trình bị ngắt đột ngột giữa lúc ghi không bao giờ làm hỏng các adapter cũ; cung cấp cơ chế `cleanup_stale_temp_dirs` dọn sạch thư mục rác an toàn.
4. **Bắt buộc Sidecar Checksums & Kiểm tra tương thích nghiêm ngặt:**
   - Bắt buộc tệp `checksums.sha256` sidecar bảo vệ toàn bộ tệp bao gồm cả `paxg_manifest.json`, fail-fast `FileNotFoundError` nếu thiếu (loại bỏ hoàn toàn rủi ro hạ cấp bảo mật ngầm).
   - `TimesFM3Predictor.load_adapter()` từ chối adapter không có manifest; tách riêng `load_raw_adapter_unsafe()`.
   - Kiểm tra tương thích chặt chẽ trước suy luận (`forecast_request`): kiểm tra độ dài ngữ cảnh, số lượng và thứ tự cột đặc trưng, revision của base model.
5. **Tích hợp Overlap Detection vào Backtest P2 từng Fold:**
   - `manifest.check_in_sample_overlap(fold_eval_start, fold_eval_end)` tích hợp trực tiếp vào vòng lặp đánh giá fold trong `BacktestEngine`.
   - Phạm vi in-sample bao quát cả dải dừng sớm: `in_sample_start = min(train_start, val_start)` và `in_sample_end = max(train_end, val_end)`.
   - Gắn cờ cảnh báo `is_in_sample=True` và `IN-SAMPLE OVERLAP DETECTED` cho fold bị trùng, tổng hợp trong `ScoreReport.metadata`.
6. **Bảo toàn Base Model Invariant:**
   - Kiểm thử `test_base_restoration_invariant`: Chuỗi chuyển đổi $\text{Base} \rightarrow \text{Adapter A} \rightarrow \text{Adapter B} \rightarrow \text{Base}$ cho đầu ra trùng khớp tuyệt đối ($|F_0 - F_0'| = 0.00e+00$), các tensor trọng số gốc khớp từng bit, sạch 100% các khóa LoRA.
7. **Bằng chứng thực nghiệm trên RTX 5060 Ti:**
   - Kịch bản `scripts/run_p3_training.py` đã huấn luyện thành công 2 adapter thực tế:
     - 1h (`horizon = 24`): `paxg_1h_r4_setB_seed42_20260905_171246` (`val_loss = 0.011693`, 95.1s).
     - 4h (`horizon = 6`): `paxg_4h_r4_setB_seed42_20260905_171403` (`val_loss = 0.012610`, 70.5s).
   - Toàn bộ bằng chứng được lưu tại `docs/paxg-lab/phases/p3_lora_proof.json`.
8. **Kiểm thử tự động:**
   - **98 tests collected: 98 passed trên workstation có GPU/cache dữ liệu (96 passed, 2 skipped trên runner CI sạch không có raw DB/snapshot cache)** (`pytest tests/paxg_lab/ src/timesfm3/ -v`), 100% pass rate.

### Tại P4:
1. **Mô hình Dữ liệu & Trạng thái Công việc (`JobSpec`, `JobStatus`, `JobPriority`):**
   - Định nghĩa trạng thái công việc tại `src/paxg_lab/queue/types.py`: `QUEUED`, `RUNNING`, `SUCCEEDED`, `FAILED`, `CANCELLED`, `INTERRUPTED`.
   - Phân cấp ưu tiên bắt buộc: `FORECAST (P1)` > `MANUAL (P2)` > `AUTO (P3)`.
   - Máy trạng thái tự động (`AutoRunState`): `SEARCHING`, `VALIDATING`, `WAITING_DATA`, `WAITING_AUDIT`, `PAUSED_ERROR`, `STOPPED`.
2. **Kho lưu trữ Hàng đợi SQLite & Chống gửi trùng (`GPUJobStorage`):**
   - Quản lý persistent trong `var/paxg_lab/paxg_lab.db` (WAL mode, busy timeout 30s).
   - Chống gửi trùng lặp (Idempotency): kiểm tra `idempotency_key`, trả về `job_id` hiện có nếu công việc cùng khóa đang `QUEUED` hoặc `RUNNING`.
   - Bảo đảm tính bất biến tối đa duy nhất 1 tiến trình GPU: transaction `BEGIN IMMEDIATE` từ chối cấp phát nếu đang có tác vụ `RUNNING`.
3. **Bộ điều phối GPU & Khóa Luân phiên 1h/4h (`GPUScheduler`):**
   - Singleton coordinator lock ngăn ngừa xung đột 2 bộ điều phối chạy đồng thời.
   - Luân phiên Round-Robin cân bằng 1h và 4h cho các công việc tự động (1h $\rightarrow$ 4h $\rightarrow$ 1h $\rightarrow$ 4h), không để đói tài nguyên một khung.
   - Dự đoán tức thời (FORECAST) chờ công việc hiện tại hoàn thành bước nhỏ rồi lập tức được ưu tiên thực thi trước toàn bộ hàng đợi.
4. **Tiến trình con GPU Riêng biệt & Thu hồi VRAM (`GPUWorker`):**
   - Mỗi công việc chạy trong tiến trình con độc lập (`sys.executable -m paxg_lab.queue.worker`).
   - Luồng nền `heartbeat_thread` cập nhật `heartbeat_at` mỗi 2s và kiểm tra cờ hủy.
   - Thu hồi hoàn toàn 100% bộ nhớ GPU khi tiến trình kết thúc (`vram_leaked = 0 bytes`).
5. **Cơ chế Heartbeat phát hiện treo & Bảo vệ Tiến trình An toàn (`process_guard`):**
   - Tự động phát hiện tiến trình bị treo khi quá hạn `heartbeat_timeout`, thu hồi tiến trình và đánh dấu `FAILED`.
   - Hàm `safe_terminate_process` bắt buộc xác minh PID và `create_time` qua `psutil`, từ chối kết thúc nếu PID bị hệ điều hành cấp lại cho ứng dụng khác.
6. **Dừng An toàn (Graceful Stop) & Phục hồi Sự cố (Crash Recovery):**
   - Lệnh `stop_auto_run()` chuyển trạng thái sang `STOPPED`, hủy toàn bộ job pending và yêu cầu job đang chạy thoát sạch tại ranh giới bước/epoch, bảo toàn checkpoint hợp lệ đã lưu.
   - Khởi động lại ứng dụng quét và phục hồi tự động các job mồ côi về `INTERRUPTED`, giải phóng khóa và không chạy đúp.
7. **Bằng chứng Thực nghiệm trên RTX 5060 Ti:**
   - Kịch bản `scripts/run_p4_queue_proof.py` đã thực thi thành công cả 6 kịch bản thực tế (ưu tiên dự đoán, luân phiên 1h/4h, treo heartbeat, dừng an toàn, ngắt đột ngột, thu hồi VRAM 0 byte).
   - Lưu trữ tại `docs/paxg-lab/phases/p4_queue_evidence.json`.
8. **Kiểm thử tự động:**
   - **63 tests collected: 63 passed** (`pytest tests/paxg_lab/ -v`), 100% pass rate.

### Tại P5:
1. **Giao diện Streamlit Tiếng Việt Đủ 5 Thẻ Chuẩn PLAN 5.1-5.6:**
   - **Thẻ 1 (Dự đoán tương lai - `tab_forecast`):** Chuyển đổi linh hoạt Base TimesFM 3.0 và các LoRA adapter tương thích; thanh trượt số nến lịch sử (96-512); form điều khiển chống F5 gửi đúp; biểu đồ Plotly candlestick kết hợp đường trung vị $q_{50}$ và dải bất định danh nghĩa 80% $[q_{10}, q_{90}]$; bảng kết quả chi tiết đúng 24 nến (1h) hoặc 6 nến (4h) kèm chênh lệch giá và %; lưu và xem lịch sử dự đoán.
   - **Thẻ 2 (Backtest & Đánh giá - `tab_backtest`):** Chọn mô hình đánh giá (Base vs LoRA), chọn fold kiểm định (Fold 1, Fold 2, Fold 3); phát hiện cảnh báo rò rỉ in-sample; thẻ tóm tắt Score v1, MAE, Pinball, Direction Accuracy; bảng so sánh trực diện 3 bên (Candidate vs Base vs Naive); xuất báo cáo JSON và CSV.
   - **Thẻ 3 (Huấn luyện thủ công LoRA - `tab_training`):** Bảng điều khiển siêu tham số khóa chặt theo quy tắc PLAN 3.2 (context length 256/512, feature set A/B/C, rank 4/8/16, alpha 8/16/32, lr, epochs 1-10, batch 1/2/4, gradient accumulation); thẻ preflight kiểm tra VRAM & RAM an toàn trước khi khởi tạo; nút Dừng an toàn (Graceful stop); thanh tiến trình và biểu đồ loss (train loss vs validation loss) cập nhật trực quan.
   - **Thẻ 4 (Tự động tìm kiếm & Tối ưu - `tab_auto_tune`):** Hiển thị máy trạng thái `AutoRunState` (SEARCHING, VALIDATING, WAITING_DATA...); nút Bắt đầu / Tạm dừng tự động; bảng xếp hạng Leaderboard các adapter tốt nhất; nhật ký kiểm toán (Audit Trail) chi tiết từng quyết định; chức năng gửi job kiểm thử khô (dry-run test).
   - **Thẻ 5 (Quản lý Adapter - `tab_adapter_manager`):** Lọc theo khung thời gian (1h/4h); ghim bảo vệ (pin/unpin); đổi tên gợi nhớ (alias); gắn nhãn khuyến nghị (recommended); xuất gói `.zip` an toàn (loại trừ code/python); tải lên & nhập zip an toàn (chặn path traversal `..`, cấm `.py`/`.pkl`, kiểm tra SHA-256 sidecar checksums); đưa vào thùng rác (`.trash/`) và khôi phục khi cần; cơ chế Deletion Protection chống xóa adapter đang được ghim hoặc khuyến nghị.
2. **Quy ước Horizon Bắt buộc 24 Giờ & Múi giờ Việt Nam (UTC+7):**
   - Header hiển thị banner bắt buộc: 1h dự đoán 24 nến (24 giờ), 4h dự đoán 6 nến (24 giờ).
   - Mọi mốc thời gian hiển thị trên giao diện người dùng chuyển đổi chuẩn xác sang múi giờ Việt Nam (`Asia/Ho_Chi_Minh` UTC+7).
3. **Phân tách Tuyệt đối Streamlit và Tính toán GPU:**
   - Streamlit không bao giờ nạp trọng số mô hình hoặc chạy autograd trực tiếp.
   - Toàn bộ tác vụ suy luận, huấn luyện, backtest được gửi dưới dạng `JobSpec` vào SQLite job queue (`var/paxg_lab/paxg_lab.db`) và được điều phối bởi `GPUScheduler` độc lập.
   - Tránh hoàn toàn xung đột VRAM, crash tiến trình web hay rò rỉ bộ nhớ.
4. **Cơ chế Chống Gửi Trùng lặp & Bảo vệ Trạng thái qua `st.form`:**
   - Mọi hành động gửi tác vụ được bọc trong `st.form` với khóa phân biệt theo session/timestamp.
   - Khi người dùng F5 hoặc mở nhiều tab trình duyệt, `idempotency_key` và trạng thái session ngăn chặn tuyệt đối việc tạo job lặp hoặc phá vỡ hàng đợi.
5. **Khởi động Một Bước & Khóa Cổng Cục bộ (`127.0.0.1:8501`):**
   - Bộ khởi động `scripts/launch_app.py`, `run_web.bat`, `run_web.ps1` chạy đồng thời `GPUScheduler` nền và giao diện Streamlit.
   - Cấu hình bắt buộc `--server.address=127.0.0.1 --server.port=8501 --server.fileWatcherType=none --browser.gatherUsageStats=false`.
6. **Kiểm thử Trình duyệt Tự động & Bằng chứng Nghiệm thu Hoàn chỉnh:**
   - Kiểm thử Playwright trên Chrome thật không đầu (`scripts/verify_p5_browser.py`) đã xác thực toàn bộ 5 thẻ, chụp ảnh minh chứng đầy đủ trong `docs/paxg-lab/phases/p5_evidence/`:
     - `p5_tab1_forecast_1h_24steps.png`
     - `p5_tab1_forecast_4h_6steps.png`
     - `p5_tab2_backtest.png`
     - `p5_tab3_training.png`
     - `p5_tab4_auto_tune.png`
     - `p5_tab5_adapter_manager.png`
     - Báo cáo JSON: `docs/paxg-lab/phases/p5_evidence/p5_browser_evidence.json`.
7. **Kiểm thử Đơn vị & Hồi quy:**
   - Bộ 9 bài kiểm thử `tests/paxg_lab/test_p5_web.py` đạt **9/9 passed (100%)**.
   - Toàn bộ 102 bài kiểm thử hệ thống `tests/paxg_lab/` đạt **102/102 passed (100%)**.

### Tại P6:
1. **Không gian Tìm kiếm Siêu tham số & Bất biến PLAN 3.6 (`src/paxg_lab/tune/space.py`):**
   - Không gian tìm kiếm Optuna TPE: `context_len` $\in [128, 256, 512]$, `lora_r` $\in [4, 8, 16]$, `lora_alpha = 2 * r` (bắt buộc), `learning_rate` $\in [1e-5, 2e-4]$ (log scale), `lora_dropout` $\in [0.0, 0.20]$, `weight_decay` $\in [0.0, 0.05]$, `history_days` $\in [180, 365, \text{"all"}]$, `feature_set` $\in [\text{"B"}, \text{"C"}]$, `modules` $\in [\text{"setB"}, \text{"all"}]$.
   - Quy ước Horizon bắt buộc theo khung: 1h $\rightarrow$ 24 nến; 4h $\rightarrow$ 6 nến.
   - Batch size hiệu dụng cố định = 16 (`batch_size=2`, `gradient_accumulation_steps=8`).
   - Hàm kiểm tra bất biến `validate_train_spec_invariants(spec)` từ chối ngay lập tức bất kỳ cấu hình vi phạm.
2. **Cơ sở dữ liệu Optuna Riêng biệt & Quy ước Tên Study (`src/paxg_lab/tune/optimizer.py`):**
   - Lưu trữ riêng tại SQLite `var/paxg_lab/optuna_studies.db` (tách biệt hoàn toàn với `paxg_lab.db` của queue/data).
   - Tên study gắn chặt theo timeframe và mã băm snapshot: `study_{timeframe}_{snapshot_hash[:8]}`.
   - Sử dụng `TPESampler(seed=42)` với `n_startup_trials=10`.
   - Giới hạn tối đa 30 trials mỗi đợt (`max_trials=30`).
   - Dừng sớm khi đình trệ (`EarlyStoppingStagnationCallback`): Tự động dừng đợt tìm kiếm nếu sau 12 trials liên tiếp không cải thiện Score v1 tối thiểu 0.5 điểm.
3. **Đánh giá Đa Hạt giống Lấy Trung vị (Multi-Seed Median):**
   - Đánh giá top cấu hình qua 3 hạt giống ngẫu nhiên `seeds = [42, 123, 2026]`.
   - Tính toán trung vị Score v1 và trung vị số epoch tối ưu qua 3 hạt giống, loại bỏ nhiễu ngẫu nhiên từ quá trình khởi tạo trọng số LoRA.
4. **Huấn luyện Lại Từ Base Model Sạch (Retrain from Base):**
   - Ứng viên chiến thắng được huấn luyện lại hoàn toàn từ mô hình Base TimesFM 3.0 với epoch tối ưu đã tìm được.
   - Tuyệt đối không warm-start hay tái sử dụng adapter từ các trials thăm dò.
5. **Bảo vệ Tuyệt đối Tập Kiểm chứng Kín (Zero Leakage):**
   - Tập kiểm chứng khóa kín 90 ngày cuối `[test_start, test_end)` tuyệt đối không bao giờ được chạm tới trong toàn bộ quá trình tìm kiếm siêu tham số, huấn luyện hay dừng sớm.
   - Chỉ được mở ra duy nhất một lần (`run_locked_verification`) trên ứng viên được khóa sau khi vượt qua vòng multi-seed median.
6. **Phân vùng Khối Độc lập 24 Giờ & Block Bootstrap 1000 Mẫu (`src/paxg_lab/tune/bootstrap.py`):**
   - Chia tập kiểm chứng kín thành các khối độc lập không chồng lấn dài 24 giờ:
     - 1h: mỗi khối 24 nến.
     - 4h: mỗi khối 6 nến.
   - Kiểm tra ngưỡng an toàn thống kê: Bắt buộc tối thiểu 20 khối độc lập dài 24 giờ.
   - Thực hiện 1,000 lần tái chọn mẫu khối ngẫu nhiên có hoàn lại (Block Bootstrap Resampling) để tính khoảng tin cậy 95% (CI 95%) của chênh lệch MAE so với Base: $\Delta_{\text{MAE}} = \text{MAE}_{\text{base}} - \text{MAE}_{\text{cand}}$.
7. **Người Gác cổng Công nhận LoRA Thắng với 7 Tiêu chí Nghiêm ngặt (`src/paxg_lab/tune/gatekeeper.py`):**
   - Tiêu chí 1: Score v1 $\ge$ điểm hiện tại + 2.0.
   - Tiêu chí 2: MAE tổng thể $\le 0.99 \times \text{MAE}_{\text{base}}$ và $\le 0.99 \times \text{MAE}_{\text{naive}}$.
   - Tiêu chí 3: Fold tệ nhất $\le 1.05 \times \text{MAE}_{\text{base}}$.
   - Tiêu chí 4: Độ bao phủ dải bất định 80% $[q_{10}, q_{90}] \in [0.65, 0.95]$.
   - Tiêu chí 5: Đủ tối thiểu 20 khối kiểm chứng độc lập dài 24 giờ.
   - Tiêu chí 6: Cận dưới khoảng tin cậy 95% Bootstrap $> 0$ ($\Delta_{\text{MAE}} > 0$ có ý nghĩa thống kê).
   - Tiêu chí 7: Adapter nạp thử thành công và bản sao lưu zip nguyên vẹn.
   - Nếu bất kỳ tiêu chí nào thất bại: Từ chối thay thế, giữ nguyên bản adapter khuyến nghị hiện tại, ghi audit log chi tiết và tự động chuyển trạng thái sang `WAITING_DATA`.
8. **Quy trình Tối ưu Tự động Hoàn chỉnh (`src/paxg_lab/tune/protocol.py`):**
   - Điều phối tuần tự: `SEARCHING` $\rightarrow$ `VALIDATING` $\rightarrow$ `WAITING_DATA` (nếu từ chối) hoặc `WAITING_AUDIT` (nếu vượt qua kiểm chứng thành công).
   - Hỗ trợ dừng an toàn (`is_cancelled_func`) tại ranh giới từng trial và từng fold.
9. **Tích hợp Giao diện Web Thẻ 4 (`tab_auto_tune.py`):**
   - Hiển thị trực quan dữ liệu trials thật từ `var/paxg_lab/optuna_studies.db`.
   - Hiển thị bảng tổng hợp trials, biểu đồ tiến trình tối ưu hóa và nhật ký kiểm toán (Audit Trail) từ `var/paxg_lab/audit_reports/`.
10. **Kiểm chứng lại sau review PR #6 (07/09/2026):**
    - Đã sửa Base/candidate dùng sai split ở vòng sau, race STOPPED, phục hồi sai snapshot, preflight gap trước khi tiêu thụ kiểm chứng và nhánh đọc JSON locked range.
    - Chạy thật phát hiện và sửa thêm hoàn thành Optuna trial sai kiểu đối tượng, cùng lỗi sampler lặp đề xuất giữa các job.
    - Mã nguồn `d802a42`: **223/223 test PAXG Lab + TimesFM3 qua**, CI GitHub xanh.
    - GPU RTX 5060 Ti chạy trọn protocol: 2 cấu hình khác nhau, 3 seed × 3 fold mỗi cấu hình, final fit, 90 khối locked verification. Score -1,929513; bootstrap CI [-1,442044; 0,366126]; từ chối ứng viên đúng và về WAITING_DATA.
    - Đây là replay lịch sử trong kho riêng với ngân sách học giảm, không cập nhật winner của ứng dụng. Không chứng minh cải thiện dự báo hoặc vận hành dài ngày.
    - Bằng chứng cũ đã được thay thế; xem `phases/P6.md` và `phases/p6_evidence/p6_real_run_evidence.json`, kèm audit và ledger. P7 chưa thực hiện; PR chưa merge.

---

## 3. Lệnh tiếp tục cho giai đoạn tiếp theo (P7)

Sau khi nghiệm thu P6, chuyển sang P7 bằng lệnh:

```text
/goal Đọc bộ tài liệu docs/paxg-lab và thực hiện P7: kiểm tra chạy dài, rò rỉ bộ nhớ, lỗi mạng, dừng/phục hồi, bảo toàn LoRA thắng và khôi phục bản sao. Chạy soak test tối thiểu 6 giờ, sửa lỗi trong phạm vi, hoàn thiện hướng dẫn và báo cáo nghiệm thu cuối.
```

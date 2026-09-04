# Ứng dụng dự báo XAUUSDT bằng TimesFM 3.0

## Tóm tắt

- Xây web app tiếng Việt bằng Streamlit + Plotly, chạy riêng trên `localhost` của từng máy Windows 11.
- Chỉ nghiên cứu: không API key, không tài khoản Binance, không WebSocket riêng tư và tuyệt đối không có chức năng đặt lệnh.
- Dữ liệu lấy từ public Binance USDⓈ-M Futures cho đúng `XAUUSDT`; hỗ trợ hai chế độ:
  - `1h`: dự báo 24 nến, tương đương 24 giờ.
  - `4h`: dự báo 6 nến, tương đương 24 giờ.
- Người dùng chọn `Base` hoặc một LoRA cụ thể khi dự báo. LoRA kém vẫn được giữ nhưng gắn nhãn “Không khuyến nghị”.
- TimesFM 3.0 có khoảng 0,3 tỷ tham số, checkpoint F32 khoảng 1,32 GB và giấy phép chỉ cho phi thương mại/nghiên cứu; cố định revision `43046b85ec22d584a13f8098c2ed39c889e129c2` để kết quả và LoRA không lệch phiên bản. [Model card chính thức](https://huggingface.co/google/timesfm-3.0-pytorch)

## Thay đổi chính

### Web và dữ liệu

- Giao diện gồm ba phần:
  - **Dự báo:** chọn 1h/4h, Base/LoRA, xem nến lịch sử, đường dự báo median và vùng q10–q90.
  - **Backtest:** so sánh Base, LoRA và baseline “giữ nguyên giá đóng cửa gần nhất”.
  - **Huấn luyện:** một nút chạy chuỗi tải dữ liệu → train LoRA 1h → backtest → train LoRA 4h → backtest.
- Chỉ gọi các endpoint đọc công khai:
  - `GET /fapi/v1/exchangeInfo`
  - `GET /fapi/v1/time`
  - `GET /fapi/v1/klines`
- Luôn lọc chính xác bản ghi `XAUUSDT`, kiểm tra trạng thái hợp đồng, phân trang tối đa 1.000 nến/lần, retry có giới hạn khi gặp 429/5xx và không tạo dữ liệu giả khi Binance lỗi. [Binance connector chính thức](https://github.com/binance/binance-futures-connector-python/blob/main/binance/um_futures/market.py)
- Chỉ giữ nến đã đóng theo giờ máy chủ Binance; loại trùng, sai thứ tự, NaN, OHLC không hợp lệ và khoảng trống thời gian.
- Loại dữ liệu trước ngày XAUUSDT mở công khai `2026-01-05`. Snapshot hiện tại còn 5.815 nến 1h và 1.453 nến 4h, đều liên tục. XAUUSDT TradFi Perpetual hoạt động 24/7 nên có thể gắn đúng 24/6 mốc dự báo tương lai. [Thông báo Binance](https://www.binance.com/en-IN/support/announcement/detail/ecf7318c0d434c339e80878588e700d0)
- TimesFM nhận `close` làm mục tiêu; `open`, `high`, `low`, `volume` là covariate lịch sử. Không dùng bất kỳ dữ liệu tương lai nào.

### LoRA và chống overfit

- Tách phần tính toán bên trong `decode()` thành `_decode_impl()` có gradient; `decode()` công khai vẫn giữ `no_grad`, nên hành vi inference hiện tại không đổi.
- Dùng Hugging Face PEFT trực tiếp với mô hình PyTorch v3:
  - Target: `query_proj`, `value_proj`.
  - Rank 4, alpha 8, dropout 0,05, bias không train.
  - AdamW, learning rate `1e-4`, weight decay `0,01`, gradient clip `1,0`.
  - Quantile pinball loss trên đủ chín quantile; không chỉ tối ưu median.
  - Tối đa 8 epoch, early stopping patience 2, seed 42, không tự dò hyperparameter.
- PEFT hỗ trợ mô hình `torch.nn.Module` tùy chỉnh nên không cần chuyển TimesFM 3.0 thành Transformers model. [Tài liệu PEFT](https://huggingface.co/docs/peft/developer_guides/custom_models)
- Chia dữ liệu theo thời gian `70% train / 15% validation / 15% test`; cửa sổ mục tiêu cách nhau đúng một horizon để giảm trùng lặp. Early stopping chỉ nhìn validation; phần test chỉ được dùng sau khi train kết thúc.
- Backtest báo cáo:
  - MAE chính, RMSE, sMAPE.
  - Độ đúng hướng so với close cuối context.
  - Coverage của q10–q90.
  - “Điểm” = phần trăm MAE giảm so với baseline giữ nguyên giá.
- LoRA được gắn nhãn “Khuyến nghị” riêng cho từng khung giờ khi:
  - MAE thấp hơn Base ít nhất 2%.
  - MAE thấp hơn baseline.
  - Không có NaN/Inf và vượt qua kiểm tra tải lại.
- LoRA không đạt vẫn được lưu, hiển thị cảnh báo đỏ và vẫn cho Khầy chủ động chọn.

### Phần cứng, tiến trình và bảo vệ file

- Tự chọn profile bằng VRAM đang trống:
  - **Tiết kiệm:** từ 5 GB VRAM trống — context 256 cho cả 1h/4h, batch 1, gradient accumulation 8.
  - **Mạnh:** tổng VRAM từ 12 GB và còn trống ít nhất 10 GB — context 1.024 cho 1h, 512 cho 4h, batch 2, accumulation 4.
  - Dưới 5 GB VRAM trống: vẫn cho Base inference bằng CPU nhưng khóa LoRA training và hướng dẫn đóng ứng dụng chiếm GPU.
- Dùng FP16 autocast + GradScaler trên cả GTX 1660 Super và RTX 5060 Ti. Bộ cài Windows dùng PyTorch CUDA 12.8, tương thích cả Turing và Blackwell. [PyTorch Windows](https://pytorch.org/get-started/locally/), [ma trận CUDA/NVIDIA](https://docs.nvidia.com/datacenter/tesla/drivers/cuda-toolkit-driver-and-architecture-matrix.html)
- Mỗi job train chạy trong subprocess riêng, từng khung giờ nối tiếp nhau; subprocess kết thúc sẽ trả VRAM/RAM cho Windows. Chỉ cho một job chạy tại một thời điểm bằng lock file tạo nguyên tử.
- Không dùng DataLoader worker, không giữ tensor/loss giữa các batch; lỗi OOM hoặc mất mạng chỉ làm job thất bại, không làm sập web.
- Dữ liệu runtime nằm trong `.timesfm-gold/` và được git-ignore. CSV, trạng thái job và JSON được ghi qua file tạm rồi `os.replace`.
- Adapter lưu vào thư mục tạm cùng ổ đĩa, dùng `adapter_model.safetensors`, sau đó:
  1. Ghi manifest và SHA-256.
  2. Đóng toàn bộ file handle.
  3. Tải lại Base + LoRA trong tiến trình mới và chạy một forecast thử.
  4. Chỉ khi thành công mới đổi tên nguyên tử thành thư mục chính thức.
- Không ghi đè LoRA cũ và không tự xóa file lỗi. App quét manifest để tìm adapter, không dùng symlink hay file `latest` dễ hỏng trên Windows.
- Nút xuất ZIP chứa LoRA, manifest, metrics và snapshot dữ liệu đã train. Import chỉ nhận JSON/CSV/safetensors, giới hạn kích thước, chặn path traversal, kiểm tra checksum, model revision và feature order trước khi giải nén.
- LoRA mang giữa hai máy được; nếu máy 6 GB không đủ context lúc train, inference sẽ tự hạ context về profile tiết kiệm và hiện cảnh báo.

## Giao diện và cách triển khai

- Giữ ba module chính: giao diện Streamlit, lõi dữ liệu/mô hình, worker train/backtest; không thêm FastAPI, database, Docker, Celery hay scheduler.
- Bổ sung extra dependencies cho web/LoRA và hai script PowerShell:
  - `setup_windows.ps1`: tạo `.venv`, cài PyTorch CUDA 12.8 cùng app, kiểm tra CUDA/RAM/disk và checkpoint.
  - `run_web.ps1`: đặt UTF-8 rồi mở Streamlit tại `127.0.0.1`.
- App hiển thị GPU/profile đang dùng, dữ liệu mới nhất, checkpoint revision, job progress và cảnh báo nghiên cứu phi thương mại.
- Khi nhập LoRA từ máy khác, Base luôn được tải đúng revision ghi trong manifest; gói ZIP không chứa hoặc phân phối checkpoint gốc.

## Kiểm thử và tiêu chí hoàn thành

- Unit test parser Binance, phân trang, loại nến đang chạy, cutoff ngày mở bán, duplicate/gap và retry.
- Test bắt buộc chứng minh mọi context kết thúc trước target; train, validation và test không chồng thời gian.
- Test đúng shape dự báo `(24, 9)` cho 1h và `(6, 9)` cho 4h; không NaN và luôn `q10 ≤ median ≤ q90`.
- Test công thức metrics, baseline, điểm cải thiện và nhãn Khuyến nghị/Không khuyến nghị.
- Test crash giữa lúc lưu, checksum sai, thiếu safetensors và ZIP độc hại; mọi adapter lỗi phải bị vô hiệu hóa nhưng không bị xóa.
- Integration test ngắn một epoch xác nhận LoRA có gradient, chỉ tham số LoRA thay đổi và adapter tải lại cho kết quả nhất quán.
- Chạy setup, Base forecast và LoRA smoke test trên cả GTX 1660 Super 6 GB lẫn RTX 5060 Ti 16 GB; sau khi worker thoát, VRAM phải trở về gần mức trước job.
- Xuất LoRA ở máy RTX, nhập ở máy GTX và so sánh forecast trên cùng snapshot trong sai số số học cho phép.
- Kiểm tra mã nguồn không chứa Binance API key, phương thức POST/DELETE hoặc endpoint đặt lệnh.

## Giả định đã khóa

- Cả hai máy dùng Windows 11, chạy ứng dụng riêng trên localhost.
- Chỉ nghiên cứu cá nhân, phi thương mại; không chia sẻ checkpoint hoặc LoRA cho bên thứ ba.
- Tự động nghĩa là bấm một nút, không chạy hằng ngày/hằng tuần.
- Dự báo là giá đóng cửa XAUUSDT; OHLCV chỉ là dữ liệu đầu vào lịch sử.
- Base và từng LoRA luôn có thể được chọn thủ công; adapter kém không bị xóa.

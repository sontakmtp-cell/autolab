# Ứng dụng dự báo XAUUSDT bằng TimesFM 3.0

## 1. Mục tiêu dự án

Xây dựng web app nghiên cứu cá nhân bằng **Streamlit + Plotly** để dự báo giá đóng cửa hợp đồng tương lai `XAUUSDT` trên Binance USDⓈ-M Futures bằng **TimesFM 3.0**, hỗ trợ cả Base model và LoRA.

Phạm vi bị khóa:

- Chạy riêng trên `localhost` của Windows 11.
- Chỉ dùng public Binance Futures API, không API key, không tài khoản Binance, không private WebSocket.
- Tuyệt đối không có chức năng đặt lệnh, quản lý vị thế hoặc kết nối tài khoản giao dịch.
- Hai timeframe:
  - `1h`: dự báo 24 nến = 24 giờ.
  - `4h`: dự báo 6 nến = 24 giờ.
- Target là `close`.
- `open`, `high`, `low`, `volume` chỉ được dùng làm covariate lịch sử.
- Người dùng luôn có thể chọn `Base` hoặc một LoRA cụ thể khi dự báo.
- LoRA kém không bị xóa, chỉ gắn nhãn **Không khuyến nghị**.
- TimesFM 3.0 cố định revision `43046b85ec22d584a13f8098c2ed39c889e129c2` để tránh sai lệch giữa Base và LoRA.
- Không phân phối checkpoint gốc TimesFM trong ZIP export.

Tham khảo:

- [TimesFM 3.0 model card](https://huggingface.co/google/timesfm-3.0-pytorch)
- [Hugging Face PEFT custom models](https://huggingface.co/docs/peft/developer_guides/custom_models)
- [Binance Futures connector](https://github.com/binance/binance-futures-connector-python/blob/main/binance/um_futures/market.py)

---

# 2. Nguyên tắc triển khai

Dự án phải được triển khai tuần tự theo các phase bên dưới. **Không được nhảy phase nếu Gate của phase trước chưa đạt.**

Mỗi phase phải:

1. Chỉ thay đổi đúng phạm vi cần thiết.
2. Có test hoặc cách kiểm chứng rõ ràng.
3. Không làm hỏng hành vi đã hoàn thành ở phase trước.
4. Không thêm kiến trúc ngoài phạm vi như FastAPI, database, Docker, Celery hoặc scheduler.
5. Ưu tiên logic core độc lập với Streamlit để có thể test mà không cần mở giao diện.

Thứ tự bắt buộc:

`Contracts → Data → Base Forecast → Backtest → Gradient Path → LoRA → Adapter Safety → Auto Training → UI → Windows Packaging → Acceptance`

---

# PHASE 0 — Khóa contracts, config và runtime layout

## Mục tiêu

Tạo nền móng thống nhất để những phase sau không tự định nghĩa lại symbol, horizon, feature order, checkpoint revision, thư mục runtime hoặc cấu hình LoRA.

## Phạm vi công việc

### 0.1. Central config

Tạo cấu hình duy nhất cho:

- Symbol: `XAUUSDT`.
- Timeframes: `1h`, `4h`.
- Horizon:
  - `1h → 24`.
  - `4h → 6`.
- Target: `close`.
- Historical covariates: `open`, `high`, `low`, `volume`.
- Feature order phải cố định và được ghi vào manifest.
- Model repo và revision TimesFM 3.0.
- Seed mặc định: `42`.
- Runtime root: `.timesfm-gold/`.

### 0.2. Runtime directory contract

Dùng cấu trúc tối thiểu:

```text
.timesfm-gold/
├── data/
├── adapters/
├── jobs/
├── cache/
├── exports/
└── locks/
```

Toàn bộ thư mục runtime phải được git-ignore.

### 0.3. Training fingerprint

Định nghĩa `training_fingerprint` để chống huấn luyện trùng.

Fingerprint phải được tạo từ ít nhất:

- model revision;
- symbol;
- timeframe;
- dataset checksum;
- train/validation/test boundaries;
- context length;
- horizon;
- feature order;
- LoRA target modules;
- rank;
- alpha;
- dropout;
- learning rate;
- weight decay;
- batch size;
- gradient accumulation;
- max epoch;
- early stopping patience;
- seed.

Serialize config theo thứ tự ổn định rồi SHA-256.

**Lock file và fingerprint có nhiệm vụ khác nhau:**

- Lock file: chống hai job chạy đồng thời.
- Fingerprint: chống train lại đúng cùng một cấu hình/dataset đã hoàn thành.

### 0.4. Atomic file helper

Tạo helper ghi JSON/CSV/status qua file tạm cùng ổ đĩa rồi `os.replace`.

Không ghi trực tiếp vào file chính.

## Deliverables

- Central config.
- Runtime-path helper.
- Atomic write helper.
- Training fingerprint helper.
- Unit tests cho config/fingerprint/atomic write.

## Gate PHASE 0

Chỉ qua phase khi:

- cùng input luôn tạo cùng fingerprint;
- đổi bất kỳ thông số quan trọng nào thì fingerprint đổi;
- runtime path không nằm ngoài `.timesfm-gold/`;
- atomic write không để lại file chính bị cắt dở khi giả lập lỗi.

---

# PHASE 1 — Binance public data pipeline

## Mục tiêu

Có pipeline dữ liệu XAUUSDT sạch, xác định được snapshot chính xác và tuyệt đối không dùng dữ liệu tương lai.

## Phạm vi công việc

### 1.1. Endpoint được phép

Chỉ gọi public read endpoints:

- `GET /fapi/v1/exchangeInfo`
- `GET /fapi/v1/time`
- `GET /fapi/v1/klines`

Không tồn tại API key, POST, DELETE hoặc endpoint đặt lệnh trong source code.

### 1.2. Symbol validation

Trước khi tải klines:

- tìm chính xác `XAUUSDT` trong `exchangeInfo`;
- xác minh contract còn hoạt động;
- không fallback sang symbol gần giống;
- Binance lỗi thì báo lỗi thật, không tạo dữ liệu giả.

### 1.3. Pagination và retry

- Tối đa 1.000 nến/request.
- Pagination theo thời gian.
- Retry có giới hạn cho 429/5xx.
- Có backoff.
- Không retry vô hạn.

### 1.4. Closed-candle filter

Dùng `/fapi/v1/time` làm server clock.

Chỉ giữ nến đã đóng hoàn toàn.

Không dựa vào giờ máy Windows để quyết định nến đã đóng hay chưa.

### 1.5. Data validation

Loại hoặc fail rõ ràng khi gặp:

- duplicate timestamp;
- timestamp sai thứ tự;
- NaN/Inf;
- OHLC phi logic;
- volume âm;
- gap timeframe;
- record trước ngày XAUUSDT mở công khai `2026-01-05`.

Không tự nội suy gap cho training/backtest.

### 1.6. Snapshot

Mỗi snapshot phải có metadata:

- symbol;
- timeframe;
- first timestamp;
- last timestamp;
- row count;
- server fetch time;
- feature order;
- SHA-256 checksum.

Snapshot 1h và 4h độc lập.

## Deliverables

- Binance public client.
- Kline downloader/paginator.
- Validator.
- Snapshot writer/reader.
- Unit tests.

## Gate PHASE 1

Phải chứng minh:

- không có nến đang chạy;
- không duplicate/gap;
- checksum ổn định trên cùng snapshot;
- parser đúng cả 1h và 4h;
- Binance lỗi không sinh dữ liệu giả;
- code không chứa trading endpoint/API key.

---

# PHASE 2 — TimesFM Base inference

## Mục tiêu

Base TimesFM phải dự báo ổn định trước khi đụng tới LoRA.

## Phạm vi công việc

### 2.1. Pin checkpoint

Luôn tải đúng revision:

`43046b85ec22d584a13f8098c2ed39c889e129c2`

App phải hiển thị revision thực tế đang dùng.

### 2.2. Hardware profile

Tự chọn profile dựa trên VRAM thực tế đang trống.

**Tiết kiệm:**

- yêu cầu từ khoảng 5 GB VRAM trống;
- context 256 cho 1h/4h;
- batch 1;
- gradient accumulation 8 ở training phase sau.

**Mạnh:**

- GPU tổng từ 12 GB và còn ít nhất 10 GB trống;
- context 1.024 cho 1h;
- context 512 cho 4h;
- batch 2;
- accumulation 4 ở training phase sau.

**Dưới 5 GB VRAM trống:**

- Base inference được phép fallback CPU;
- LoRA training bị khóa;
- UI phải giải thích lý do.

### 2.3. Base forecast API

Core forecast phải trả về cấu trúc chuẩn, độc lập Streamlit:

- timestamps dự báo;
- 9 quantiles;
- median;
- q10;
- q90;
- metadata model/profile/context.

Shape bắt buộc:

- `1h → (24, 9)`.
- `4h → (6, 9)`.

### 2.4. Validation

Forecast phải:

- không NaN/Inf;
- có đúng horizon;
- timestamps liên tiếp đúng timeframe;
- `q10 ≤ median ≤ q90` ở mọi bước.

## Deliverables

- Model loader.
- Hardware profiler.
- Base forecast service.
- Tests shape/quantile/revision/profile.

## Gate PHASE 2

Base forecast phải chạy thành công trên snapshot thật cho cả 1h và 4h trước khi xây backtest hoặc LoRA.

---

# PHASE 3 — Backtest và baseline chuẩn

## Mục tiêu

Khóa hệ thống đánh giá trước khi training để tránh thay metric sau khi nhìn thấy kết quả LoRA.

## Phạm vi công việc

### 3.1. Time split

Chia dữ liệu theo thời gian:

- 70% train;
- 15% validation;
- 15% test.

Không shuffle.

### 3.2. Leakage protection

Bắt buộc:

- mọi context kết thúc trước target;
- target windows không overlap giữa các split;
- các cửa sổ evaluation cách nhau ít nhất một horizon;
- purge/embargo tại ranh giới split nếu context hoặc horizon có nguy cơ chạm sang split khác;
- scaler/normalizer nếu có fit thì chỉ fit trên train;
- validation chỉ phục vụ model selection/early stopping;
- test bị khóa cho tới khi training kết thúc.

### 3.3. Baseline

Baseline chính:

> mọi bước tương lai bằng close cuối cùng của context.

Baseline phải dùng đúng cùng evaluation windows với Base/LoRA.

### 3.4. Metrics

Báo cáo ít nhất:

- MAE — metric chính;
- RMSE;
- sMAPE;
- Directional Accuracy so với close cuối context;
- coverage q10–q90.

Điểm cải thiện:

```text
score_pct = (baseline_MAE - model_MAE) / baseline_MAE * 100
```

### 3.5. Recommendation rule

LoRA chỉ được gắn **Khuyến nghị** riêng theo timeframe khi:

- MAE tốt hơn Base ít nhất 2%;
- MAE tốt hơn baseline;
- không NaN/Inf;
- adapter vượt qua reload smoke test ở PHASE 6.

Không đạt thì **Không khuyến nghị** nhưng không xóa.

## Deliverables

- Split/window builder.
- Leakage checks.
- Baseline evaluator.
- Metrics engine.
- Base backtest report cho 1h và 4h.

## Gate PHASE 3

Trước khi train LoRA phải có:

- test chứng minh split không overlap;
- test context/target không leakage;
- metric unit tests;
- Base + baseline backtest chạy được cho cả 1h và 4h.

---

# PHASE 4 — Mở gradient path cho TimesFM 3.0

## Mục tiêu

Cho phép training LoRA mà không thay đổi hành vi inference công khai hiện tại.

## Phạm vi công việc

### 4.1. Surgical model change

Tách phần computation bên trong `decode()` thành `_decode_impl()` có gradient.

- `_decode_impl()` không bọc `no_grad`.
- `decode()` công khai vẫn giữ `no_grad` như trước và gọi `_decode_impl()`.

Không refactor model ngoài phần bắt buộc.

### 4.2. Regression tests

Trước và sau thay đổi:

- cùng input;
- cùng checkpoint;
- cùng seed/config;
- public inference phải cho kết quả tương đương trong tolerance cho phép.

### 4.3. Gradient smoke test

Test một forward/backward nhỏ để chứng minh:

- gradient tới được target modules;
- không gradient vào frozen parameters khi PEFT được áp dụng ở phase sau.

## Deliverables

- Minimal TimesFM model patch.
- Regression tests.
- Gradient-path test.

## Gate PHASE 4

Không được sang LoRA trainer nếu public Base inference thay đổi ngoài tolerance hoặc gradient path chưa được chứng minh.

---

# PHASE 5 — LoRA trainer và chống overfit

## Mục tiêu

Huấn luyện LoRA an toàn, reproducible và chỉ cập nhật đúng adapter parameters.

## Phạm vi công việc

### 5.1. PEFT config cố định

Dùng Hugging Face PEFT trực tiếp với TimesFM PyTorch module.

Thông số mặc định:

- target modules: `query_proj`, `value_proj`;
- rank: `4`;
- alpha: `8`;
- dropout: `0.05`;
- bias: không train;
- optimizer: AdamW;
- learning rate: `1e-4`;
- weight decay: `0.01`;
- gradient clip: `1.0`;
- max epoch: `8`;
- early stopping patience: `2`;
- seed: `42`;
- không tự động dò hyperparameter.

### 5.2. Loss

Dùng quantile pinball loss trên đủ 9 quantiles.

Không chỉ tối ưu median.

### 5.3. Parameter safety

Trước training phải snapshot danh sách trainable parameters.

Bắt buộc xác minh:

- Base parameters frozen;
- chỉ LoRA parameters trainable;
- không cùng parameter xuất hiện hai lần trong optimizer groups;
- không duplicated parameter ID;
- số lượng trainable parameters được log vào manifest/job report.

### 5.4. Training loop

- FP16 autocast + GradScaler trên CUDA nếu phần cứng hỗ trợ profile tương ứng.
- Không dùng DataLoader worker.
- Không giữ tensor/loss graph giữa batch.
- `zero_grad(set_to_none=True)`.
- gradient clipping.
- early stopping chỉ nhìn validation loss.
- test set hoàn toàn không được đọc để quyết định epoch/model.

### 5.5. OOM protection

OOM phải:

- đánh dấu job failed rõ ràng;
- giải phóng object lớn;
- không làm sập Streamlit;
- không publish adapter dở;
- không âm thầm đổi hyperparameter giữa job.

### 5.6. Subprocess isolation

Mỗi train job chạy trong subprocess riêng.

Train 1h và 4h chạy tuần tự, không song song.

Khi subprocess kết thúc, VRAM/RAM phải được giải phóng gần mức trước job.

## Deliverables

- LoRA config builder.
- Trainer.
- Early stopping.
- Parameter audit.
- One-epoch integration test.

## Gate PHASE 5

Phải chứng minh:

- LoRA nhận gradient;
- chỉ LoRA parameters thay đổi;
- Base parameters không đổi;
- optimizer không chứa parameter trùng;
- early stopping không dùng test;
- one-epoch smoke train chạy được.

---

# PHASE 6 — Adapter registry, chống mất LoRA và file corruption

## Mục tiêu

Không có trường hợp adapter lỗi được xem là hợp lệ hoặc adapter cũ bị ghi đè/mất ngoài ý muốn.

## Phạm vi công việc

### 6.1. Immutable adapter directories

Mỗi LoRA có directory riêng, tên chứa tối thiểu:

- timeframe;
- timestamp;
- short fingerprint.

Không ghi đè directory cũ.

### 6.2. Temporary staging

Sau training, ghi vào thư mục tạm cùng ổ đĩa.

Bắt buộc có:

- `adapter_model.safetensors`;
- adapter config cần thiết;
- `manifest.json`;
- `metrics.json`;
- dataset metadata/checksum.

### 6.3. Manifest

Manifest phải chứa ít nhất:

- schema version;
- created_at;
- symbol;
- timeframe;
- TimesFM model repo/revision;
- feature order;
- context/horizon;
- dataset checksum;
- split boundaries;
- training fingerprint;
- LoRA hyperparameters;
- seed;
- trainable parameter count;
- best epoch;
- validation metrics;
- test metrics sau training;
- recommendation status;
- SHA-256 của adapter files.

### 6.4. Reload verification

Trước khi publish adapter:

1. flush và đóng file handles;
2. tạo process mới;
3. tải Base đúng revision;
4. tải LoRA từ staging directory;
5. chạy forecast smoke test;
6. kiểm tra shape/NaN/quantile ordering;
7. kiểm checksum;
8. chỉ khi pass mới atomic rename sang directory chính thức.

### 6.5. Registry scan

App tìm LoRA bằng cách scan manifest.

Không dùng symlink.

Không dùng file `latest` làm nguồn sự thật.

Adapter lỗi:

- không xóa;
- đánh dấu invalid/quarantined;
- không cho inference mặc định;
- UI hiển thị lý do lỗi.

### 6.6. Duplicate-training protection

Trước khi train:

- scan registry;
- nếu đã tồn tại adapter hợp lệ cùng `training_fingerprint`, không train lại;
- trả về adapter hiện hữu và ghi log `SKIPPED_DUPLICATE`.

Có tùy chọn explicit force retrain ở core API nếu thực sự cần nghiên cứu lại, nhưng UI mặc định không bật.

### 6.7. Export/import ZIP

Export chứa:

- LoRA;
- manifest;
- metrics;
- snapshot dữ liệu đã train hoặc metadata theo thiết kế cuối cùng.

Không chứa TimesFM Base checkpoint.

Import:

- chỉ nhận JSON/CSV/safetensors và file cần thiết đã whitelist;
- giới hạn tổng dung lượng và số file;
- chặn absolute path, `..`, path traversal;
- không extract trước khi validate member names;
- kiểm checksum;
- kiểm model revision;
- kiểm feature order;
- không ghi đè LoRA hiện có.

## Deliverables

- Adapter registry.
- Manifest schema.
- Safe publish flow.
- Reload verifier.
- Export/import.
- Corruption/security tests.

## Gate PHASE 6

Phải pass các test:

- crash giữa lúc save;
- checksum sai;
- thiếu safetensors;
- manifest sai revision;
- ZIP path traversal;
- adapter reload fail;
- duplicate fingerprint;
- adapter cũ không bị ghi đè.

**Auto-training chưa được triển khai trước khi Gate này đạt.**

---

# PHASE 7 — One-click auto training + backtest pipeline

## Mục tiêu

Tạo chuỗi tự động bấm một nút nhưng vẫn dùng lại các module đã kiểm chứng ở phase trước.

## Flow bắt buộc

```text
Acquire global atomic job lock
        ↓
Download/refresh data
        ↓
Validate + snapshot 1h/4h
        ↓
Compute fingerprint 1h
        ↓
Skip duplicate hoặc train LoRA 1h
        ↓
Reload verify
        ↓
Backtest Base + LoRA + baseline 1h
        ↓
Publish metrics/recommendation
        ↓
Release model/process resources
        ↓
Compute fingerprint 4h
        ↓
Skip duplicate hoặc train LoRA 4h
        ↓
Reload verify
        ↓
Backtest Base + LoRA + baseline 4h
        ↓
Publish metrics/recommendation
        ↓
Release resources
        ↓
Finish job + release lock
```

## Job locking

- Chỉ một training pipeline được chạy tại một thời điểm.
- Lock tạo nguyên tử.
- Lock phải chứa PID/start time/job ID.
- Có xử lý stale lock sau khi xác minh process cũ thực sự không còn tồn tại.
- Không xóa lock đang thuộc process còn sống.

## Job status

Ghi atomic JSON gồm:

- job ID;
- state;
- current phase;
- timeframe;
- started/updated/finished time;
- progress;
- current message;
- error nếu có;
- adapter ID nếu publish thành công.

State tối thiểu:

- `QUEUED`
- `DOWNLOADING`
- `VALIDATING`
- `TRAINING_1H`
- `BACKTESTING_1H`
- `TRAINING_4H`
- `BACKTESTING_4H`
- `COMPLETED`
- `FAILED`
- `SKIPPED_DUPLICATE`

Mất mạng/OOM/lỗi adapter chỉ làm job fail, không làm web process chết.

## Deliverables

- Job runner subprocess.
- Atomic lock manager.
- Job-state writer.
- Sequential 1h → 4h orchestrator.

## Gate PHASE 7

Phải test:

- double-click không tạo hai job;
- stale lock xử lý đúng;
- duplicate fingerprint được skip;
- 1h fail thì trạng thái rõ ràng và không publish adapter dở;
- worker exit giải phóng tài nguyên;
- web vẫn sống sau worker failure.

---

# PHASE 8 — Streamlit + Plotly UI

## Mục tiêu

Đưa các core module đã ổn định lên giao diện tiếng Việt mà không nhét business logic vào UI.

## Cấu trúc giao diện

Ba khu vực chính.

### 8.1. Dự báo

Cho phép:

- chọn timeframe 1h/4h;
- chọn Base hoặc LoRA cụ thể;
- xem adapter status;
- xem historical candlestick;
- xem forecast median;
- vùng q10–q90;
- xem timestamps dự báo tương lai;
- xem model revision/context/profile.

Nếu LoRA `Không khuyến nghị`, hiển thị cảnh báo rõ nhưng vẫn cho người dùng chủ động chọn.

### 8.2. Backtest

Hiển thị cạnh nhau:

- Base;
- selected LoRA;
- persistence baseline.

Metrics:

- MAE;
- RMSE;
- sMAPE;
- Directional Accuracy;
- q10–q90 coverage;
- score cải thiện so baseline.

### 8.3. Huấn luyện

Một nút chạy auto pipeline.

Hiển thị:

- progress;
- current stage;
- timeframe đang chạy;
- duplicate skip;
- error/OOM/network message;
- kết quả adapter mới;
- recommendation status.

### 8.4. System status

App phải hiển thị:

- GPU;
- VRAM total/free;
- hardware profile;
- CPU fallback nếu có;
- latest data timestamp;
- TimesFM revision;
- training capability enabled/disabled;
- cảnh báo nghiên cứu phi thương mại.

### 8.5. Adapter management

Cho phép:

- xem danh sách LoRA;
- xem manifest/metrics cơ bản;
- export ZIP;
- import ZIP;
- phân biệt Recommended / Not Recommended / Invalid.

Không có nút tự xóa adapter trong phiên bản đầu.

## Deliverables

- Streamlit app.
- Plotly chart.
- Forecast tab.
- Backtest tab.
- Training tab.
- Adapter import/export UI.

## Gate PHASE 8

UI phải chỉ gọi core APIs và không tự thực hiện split/train/metric logic riêng.

Base forecast phải dùng được ngay cả khi chưa có LoRA.

---

# PHASE 9 — Windows setup và launcher

## Mục tiêu

Cho phép cài và chạy ứng dụng ổn định trên hai máy Windows 11 mục tiêu.

## Phạm vi công việc

### 9.1. `setup_windows.ps1`

Script phải:

- kiểm tra Python phù hợp;
- tạo `.venv`;
- cài dependencies;
- cài PyTorch CUDA 12.8 theo cấu hình dự án;
- kiểm tra `torch.cuda.is_available()`;
- hiển thị GPU/VRAM;
- kiểm RAM/disk tối thiểu;
- tải/kiểm TimesFM revision nếu cần;
- tạo runtime directory;
- không yêu cầu Binance API key.

### 9.2. `run_web.ps1`

- activate `.venv`;
- đặt UTF-8;
- chạy Streamlit bind `127.0.0.1`;
- không expose `0.0.0.0` mặc định.

### 9.3. GPU compatibility targets

Smoke test trên:

- GTX 1660 Super 6 GB;
- RTX 5060 Ti 16 GB.

Dùng FP16 autocast + GradScaler ở training khi hợp lệ với runtime.

Nếu GPU 6 GB không đủ context của adapter từng được train ở máy mạnh, inference được phép hạ về profile tiết kiệm và phải hiển thị cảnh báo.

## Deliverables

- `setup_windows.ps1`.
- `run_web.ps1`.
- Environment diagnostics.
- README hướng dẫn ngắn.

## Gate PHASE 9

Một máy Windows sạch phải có thể đi từ setup → run web → Base forecast mà không chỉnh source code thủ công.

---

# PHASE 10 — Full acceptance, cross-machine và release gate

## Mục tiêu

Xác minh toàn bộ hệ thống trước khi coi phiên bản nghiên cứu đầu tiên hoàn thành.

## 10.1. Data tests

- parser Binance;
- pagination;
- retry;
- closed candle;
- cutoff `2026-01-05`;
- duplicate;
- gap;
- NaN/Inf;
- OHLC validation;
- deterministic snapshot checksum.

## 10.2. Leakage tests

Bắt buộc chứng minh bằng code/tests:

- context luôn kết thúc trước target;
- train/validation/test không overlap target time;
- purge/embargo hoạt động;
- test set không được dùng trong early stopping;
- không dùng future covariate.

## 10.3. Forecast tests

- `1h → (24, 9)`;
- `4h → (6, 9)`;
- không NaN/Inf;
- timestamps đúng;
- `q10 ≤ median ≤ q90`.

## 10.4. Backtest tests

- MAE;
- RMSE;
- sMAPE;
- Directional Accuracy;
- interval coverage;
- baseline;
- score formula;
- recommendation rule.

## 10.5. LoRA tests

- gradient tồn tại;
- Base frozen;
- chỉ LoRA thay đổi;
- optimizer không có duplicate parameters;
- early stopping;
- reload consistency;
- fingerprint duplicate skip.

## 10.6. Artifact safety tests

- crash lúc ghi;
- checksum sai;
- safetensors thiếu/hỏng;
- manifest sai revision;
- feature order mismatch;
- malicious ZIP/path traversal;
- invalid adapter bị vô hiệu hóa nhưng không bị tự xóa.

## 10.7. Process/memory tests

Trên cả GTX 6 GB và RTX 16 GB:

- Base forecast;
- one-epoch LoRA smoke test;
- worker subprocess exit;
- VRAM sau worker trở về gần mức trước job;
- chạy nhiều job tuần tự không tăng VRAM/RAM liên tục;
- OOM không làm Streamlit chết.

## 10.8. Cross-machine LoRA test

1. Train/export LoRA trên RTX 5060 Ti.
2. Import trên GTX 1660 Super.
3. Load đúng Base revision.
4. Forecast cùng snapshot/context.
5. So sánh output trong sai số floating-point cho phép.

## 10.9. Security/scope audit

Search toàn repository để đảm bảo không tồn tại:

- Binance API key/secret;
- order endpoint;
- POST/DELETE Binance trading requests;
- private account websocket;
- chức năng đặt lệnh;
- checkpoint TimesFM gốc nằm trong export package.

## Final Gate

Phiên bản nghiên cứu chỉ được coi là hoàn thành khi:

- toàn bộ Gate PHASE 0–9 đều pass;
- full test suite pass;
- smoke test hai GPU pass;
- cross-machine adapter test pass;
- không phát hiện data leakage;
- không phát hiện Base parameter bị train;
- không phát hiện training duplicate ngoài explicit force retrain;
- adapter hỏng không thể được chọn như adapter hợp lệ;
- web vẫn hoạt động sau lỗi network/OOM/worker crash.

---

# 3. Kiến trúc cuối cùng bị khóa

Giữ kiến trúc đơn giản gồm ba lớp chính:

```text
Streamlit / Plotly UI
        ↓
Core data + forecast + backtest + adapter registry
        ↓
Isolated training/backtest worker subprocess
```

Không thêm:

- FastAPI;
- PostgreSQL/SQLite;
- Docker;
- Celery;
- Redis;
- scheduler;
- cloud deployment;
- trading engine.

Tự động trong phạm vi dự án chỉ có nghĩa là **người dùng bấm một nút để chạy pipeline**, không phải hệ thống tự train mỗi ngày/tuần.

---

# 4. Quy tắc về dữ liệu và nghiên cứu

- Chỉ dùng XAUUSDT Binance USDⓈ-M Futures.
- Không trộn spot, CFD hoặc nguồn giá vàng khác trong phiên bản đầu.
- Dự báo là giá `close` tương lai.
- OHLCV chỉ là historical input.
- Không dùng future-known feature nếu feature đó chưa thực sự biết tại forecast time.
- Snapshot training luôn được lưu/checksum để biết chính xác LoRA đã học từ dữ liệu nào.
- Backtest phải reproducible từ snapshot + manifest + model revision.
- Không thay đổi metric/rule Recommendation chỉ vì kết quả LoRA không đẹp.

---

# 5. Quy tắc về LoRA

- Base luôn được giữ nguyên.
- LoRA 1h và 4h là adapter độc lập.
- Không dùng chung adapter giữa hai timeframe.
- Không overwrite adapter cũ.
- Adapter không đạt tiêu chí vẫn giữ lại.
- Recommendation là metadata, không phải quyền xóa adapter.
- Mỗi adapter phải trace được về dataset checksum + fingerprint + TimesFM revision.
- Adapter chỉ được publish sau reload smoke test trong process mới.

---

# 6. Thứ tự làm việc dành cho AI coding agent

Agent triển khai phải làm theo từng phase và dừng ở Gate tương ứng.

Không triển khai trước code của phase tương lai nếu chưa cần thiết.

Mỗi phase nên theo chu trình:

```text
1. Đọc code liên quan
2. Ghi assumptions ngắn
3. Viết/điều chỉnh tests cho Gate
4. Implement thay đổi nhỏ nhất
5. Chạy tests
6. Sửa đến khi Gate pass
7. Ghi lại file đã thay đổi + kết quả test
8. Chỉ sau đó mới sang phase kế tiếp
```

Nếu một phase yêu cầu thay đổi TimesFM source, chỉ sửa đúng phần cần thiết và có regression test bảo vệ Base inference.

---

# 7. Definition of Done của toàn dự án

Người dùng mở `run_web.ps1`, truy cập localhost và có thể:

1. tải/refresh dữ liệu XAUUSDT thật từ Binance;
2. xem dữ liệu mới nhất và tình trạng hợp lệ;
3. chọn 1h hoặc 4h;
4. chạy Base forecast;
5. xem candle chart + median + q10–q90;
6. chạy backtest Base/baseline;
7. bấm một nút để train LoRA tuần tự 1h rồi 4h;
8. theo dõi progress mà web không bị treo/sập khi worker lỗi;
9. xem backtest LoRA và nhãn Khuyến nghị/Không khuyến nghị;
10. chọn Base hoặc bất kỳ LoRA hợp lệ nào để forecast;
11. export/import LoRA an toàn giữa hai máy;
12. không bị train trùng cùng fingerprint;
13. không mất hoặc ghi đè LoRA cũ khi crash;
14. không có data leakage giữa train/validation/test;
15. không có bất kỳ chức năng trading thật nào.

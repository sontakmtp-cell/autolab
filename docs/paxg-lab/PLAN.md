\# PAXG Forecast Lab — kế hoạch cập nhật



Khầy, chốt lại: \*\*khung 1h dự đoán 24 nến; khung 4h dự đoán 6 nến. Cả hai cùng nhìn trước 24 giờ.\*\* Kế hoạch này thay thế bản trước; số bước dự đoán, nhãn huấn luyện, backtest và kiểm chứng đều đi theo từng khung.



\## 1. Mục tiêu và kiến trúc



| Hạng mục | Quyết định |

|---|---|

| Sản phẩm | Web nghiên cứu dự đoán giá, dùng cá nhân trên Windows |

| Thị trường | Binance USDⓈ-M Futures, `PAXGUSDT`, perpetual |

| Mô hình | `google/timesfm-3.0-pytorch` |

| Khung 1h | Đầu ra 24 nến, tương đương 24 giờ |

| Khung 4h | Đầu ra 6 nến, tương đương 24 giờ |

| Giá dự đoán | Giá đóng cửa |

| Biểu đồ | Nến Nhật lịch sử, đường trung vị dự đoán và dải bất định |

| Chọn mô hình | Base hoặc LoRA tương thích |

| Tiêu chí tối ưu | Độ chính xác mức giá, chất lượng khoảng dự đoán và sự ổn định |

| Chạy tự động | Bật đến khi nhấn Dừng; nghỉ GPU khi chờ dữ liệu hoặc kiểm chứng |

| GPU đang bận | Dự đoán chờ công việc hiện tại kết thúc rồi được ưu tiên |

| Ngoài phạm vi | Đặt lệnh, khóa giao dịch, đòn bẩy, nhiều người dùng, cloud và thương mại |



Dùng \*\*Python + Streamlit + Plotly + SQLite\*\*, \*\*PEFT\*\* cho LoRA và \*\*Optuna\*\* cho tìm cấu hình.



```mermaid

flowchart LR

&#x20;   UI\["Web: 5 thẻ"] --> DB\["SQLite: cấu hình và công việc"]

&#x20;   BN\["Binance API"] --> DATA\["Kho dữ liệu và snapshot"]

&#x20;   DB --> S\["Bộ điều phối"]

&#x20;   S --> W\["Một tiến trình GPU"]

&#x20;   DATA --> W

&#x20;   W --> MODEL\["TimesFM 3.0 + LoRA tùy chọn"]

&#x20;   W --> ART\["Adapter, báo cáo và bản sao"]

&#x20;   W --> DB

```



\- Mã ứng dụng đặt trong `src/paxg\_lab/`.

\- Dữ liệu chạy đặt trong `var/paxg\_lab/`, loại khỏi Git.

\- Giao diện chỉ gửi yêu cầu và đọc tiến độ.

\- Bộ điều phối chạy riêng, không phụ thuộc phiên trình duyệt.

\- Mỗi công việc GPU dùng tiến trình con mới; kết thúc thì giải phóng tiến trình.

\- SQLite lưu dữ liệu nến và trạng thái ứng dụng; Optuna dùng tệp SQLite riêng.

\- Snapshot dùng NumPy nén, kèm thông tin kiểm chứng.

\- Không thêm React, FastAPI, Redis, Celery hay Docker.



Hai khung có riêng cấu hình, LoRA, bảng điểm, lịch sử và trạng thái tự động. Có thể bật cả hai nhưng GPU chạy luân phiên.



\*\*Điểm cần chứng minh trước:\*\* ví dụ LoRA trong repo dành cho 2.5; TimesFM 3.0 cần tích hợp đường huấn luyện riêng dựa trên mã hiện có. Không được thay tên checkpoint rồi coi như hoàn thành. \[Ví dụ LoRA hiện có](D:/AI/timesfm\_b/timesfm-forecasting/examples/finetuning/README.md).



Trọng số 3.0 có giấy phép riêng cho thử nghiệm và nghiên cứu phi thương mại, phi production; giữ đúng phạm vi cá nhân đã chốt. \[Giấy phép checkpoint](https://huggingface.co/google/timesfm-3.0-pytorch/blob/main/LICENSE).



\## 2. Dữ liệu và đầu vào mô hình



\### 2.1. Thu thập từ Binance



Dùng các endpoint:



\- `/fapi/v1/exchangeInfo`: xác minh hợp đồng và bước giá.

\- `/fapi/v1/time`: thời gian sàn.

\- `/fapi/v1/klines`: nến giao dịch 1h và 4h.

\- `/fapi/v1/markPriceKlines`: giá đánh dấu.

\- `/fapi/v1/fundingRate`: lịch sử funding.



Kiểm tra trước đó xác nhận hợp đồng đang giao dịch và lịch sử bắt đầu ngày 27/03/2025. Khi triển khai phải kiểm tra lại, không mặc định có nhiều năm dữ liệu. Tuân thủ \[giới hạn Binance Futures](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data).



Quy tắc:



\- Tải lịch sử theo trang, sau đó cập nhật phần thiếu.

\- Lưu UTC; giao diện mặc định giờ Việt Nam.

\- Chỉ nến đã đóng được dùng để dự đoán, huấn luyện và chấm điểm.

\- Nến đang chạy nếu hiển thị phải có nhãn riêng.

\- Khóa duy nhất: nguồn, mã, khung giờ, thời gian mở nến.

\- Kiểm tra OHLC, giá dương, khối lượng không âm, trùng thời gian và khoảng thiếu.

\- Loại nến niêm yết đầu tiên nếu không đủ thời lượng.

\- Không nội suy qua khoảng thiếu giá. Tải bù; nếu không được thì loại cửa sổ liên quan và báo số lượng.

\- Timeout, 429 hoặc 418 được xử lý bằng chờ và thử lại có giới hạn.

\- Mỗi đợt nghiên cứu dùng snapshot bất biến có mã băm.

\- Dữ liệu tải mới không sửa snapshot đang được sử dụng.



Không dùng open interest và tỷ lệ long/short làm đầu vào bắt buộc ở bản đầu vì lịch sử API của chúng ngắn.



\### 2.2. Ba bộ đầu vào



Mục tiêu là \*\*giá đóng cửa thực tế\*\* của PAXGUSDT Futures. Giữ cơ chế chuẩn hóa nội bộ TimesFM; không cộng dồn phân vị lợi suất để tạo khoảng dự đoán giá.



| Bộ | Thành phần |

|---|---|

| A — Giá | Chuỗi giá đóng cửa |

| B — Giá và hoạt động | A + `log1p` khối lượng USDT, `log(high/low)`, thay đổi open→close, tỷ lệ mua chủ động |

| C — Thông tin hợp đồng | B + chênh lệch mark/close và funding gần nhất đã công bố |



\- A là chuẩn đối chiếu cố định.

\- B là cấu hình mặc định ban đầu.

\- C chỉ khả dụng khi đủ lịch sử sạch trên toàn bộ phạm vi đánh giá.

\- B và C thêm giờ trong ngày, ngày trong tuần dạng sin/cos; đây là dữ liệu được phép biết trước.

\- Giá, khối lượng, mark price và funding tương lai luôn bị che.

\- Funding ghép theo thời điểm công bố, không dùng trước khi biết.

\- Khối lượng bằng 0 tạo tỷ lệ mua trung tính.

\- Adapter ghi rõ bộ đầu vào, thứ tự cột và phiên bản cách tính.

\- Thiếu đầu vào phải báo lỗi; không tự thay bộ đặc trưng của LoRA.



\### 2.3. Quy ước horizon bắt buộc



Chỉ có một nguồn cấu hình chuẩn:



```text

1h → horizon = 24

4h → horizon = 6

```



Mọi thành phần phải đọc từ quy ước này:



\- Tạo nhãn huấn luyện.

\- Cắt đầu ra TimesFM.

\- Tạo mốc thời gian tương lai.

\- Khoảng loại cửa sổ quanh ranh giới dữ liệu.

\- Trọng số chấm điểm.

\- Số khối kiểm chứng độc lập.

\- Manifest và kiểm tra tương thích LoRA.



Không rải giá trị `24` cố định trong mã dùng chung.



\## 3. Huấn luyện, backtest và tìm LoRA tốt



\### 3.1. LoRA cho TimesFM 3.0



Dùng PEFT để chèn adapter vào mô hình PyTorch hiện có. \[PEFT hỗ trợ cách tích hợp này](https://huggingface.co/docs/peft/developer\_guides/low\_level\_api).



\- Khóa trọng số base, chỉ cập nhật LoRA.

\- Mặc định gắn vào `query\_proj` và `value\_proj` của attention theo thời gian và giữa các biến.

\- Chế độ mở rộng thêm `key\_proj` và `out\_proj`.

\- Không huấn luyện toàn bộ mô hình hoặc dùng QLoRA ở bản đầu.

\- Mỗi lượt thử bắt đầu từ base cố định và adapter mới.

\- Không huấn luyện nối tiếp vô hạn từ LoRA thắng.



Mã `decode()` hiện tắt gradient. Tách phần tính toán thành đường dùng chung có thể huấn luyện, giữ `decode()` công khai ở chế độ suy luận.



Huấn luyện và dự đoán phải dùng cùng cách chuẩn bị đầu vào, che tương lai, xử lý xu hướng, ghép đầu ra và cắt horizon. Không xây hai cách dự đoán khác nhau.



Kiểm tra bắt buộc:



\- Adapter mới khởi tạo tương đương base.

\- Cập nhật làm thay đổi LoRA nhưng không thay đổi base.

\- Gradient hữu hạn và loss có thể giảm trên một tập nhỏ.

\- Lưu/nạp lại cho đầu ra tương đương.

\- Suy luận cũ không đổi ngoài dung sai số học.

\- Đầu ra đúng `(24, 9)` cho 1h và `(6, 9)` cho 4h.



TimesFM 3.0 trả 9 phân vị, trung vị ở chỉ số 4. Dải q10–q90 là khoảng danh nghĩa \*\*80%\*\*. \[Checkpoint chính thức](https://huggingface.co/google/timesfm-3.0-pytorch).



\### 3.2. Thiết lập thủ công



| Thiết lập | Mặc định | Phạm vi |

|---|---:|---|

| Context | 256 nến | 128, 256, 512 |

| Horizon | Theo khung | 1h: 24; 4h: 6 |

| LoRA rank | 4 | 2, 4, 8, 16 |

| LoRA alpha | 8 | Mặc định 2 × rank, cho sửa thủ công |

| Dropout | 0,10 | 0–0,20 |

| Learning rate | `5e-5` | `1e-5`–`3e-4` |

| Số vòng học tối đa | 5 | 1–10 |

| Batch GPU | 2 | 1, 2, 4 |

| Tích lũy gradient | 8 | 1–16 |

| Weight decay | 0,01 | 0–0,10 |

| Dừng sớm | 2 vòng không cải thiện | 1–4 |

| Gradient clipping | 1,0 | 0,5–2,0 |

| Lịch sử học | Tối đa 365 ngày | 180 ngày, 365 ngày, toàn bộ |

| Seed | 42 | Số nguyên |



Giao diện giải thích bằng tiếng Việt; thiết lập ít dùng nằm trong phần nâng cao.



Cơ chế học:



\- AdamW, learning rate cố định sau tối đa 10% bước khởi động tăng dần.

\- Mỗi vòng lấy tối đa 1.024 cửa sổ khác nhau từ phần được phép học.

\- Chỉ xáo trộn cửa sổ trong tập huấn luyện.

\- Loss kết hợp sai số trung vị và pinball loss của 9 phân vị, chia theo giá cuối đầu vào.

\- Loss tính float32; BF16 chỉ dùng sau kiểm tra số học.

\- Lưu checkpoint tốt nhất theo tập dừng sớm.

\- Thiếu dữ liệu hoặc cấu hình không vừa tài nguyên phải bị từ chối trước khi chạy.



\### 3.3. Backtest không nhìn trước



Phân biệt:



1\. \*\*Backtest adapter có sẵn:\*\* chỉ các nhãn sau thời điểm kết thúc dữ liệu học của adapter mới được gọi là ngoài mẫu.

2\. \*\*Đánh giá cấu hình huấn luyện:\*\* mỗi đoạn thời gian phải học adapter mới từ base rồi dự đoán đoạn phía sau.



Đợt đầu chia lịch sử:



\- \*\*90 ngày cuối:\*\* kiểm chứng khóa kín.

\- \*\*90 ngày trước đó:\*\* ba đoạn đánh giá, mỗi đoạn 30 ngày.

\- \*\*Trước mỗi đoạn đánh giá:\*\* phần huấn luyện; 14 ngày cuối dành cho dừng sớm.



Quy tắc ranh giới:



\- Toàn bộ nhãn của cửa sổ phải nằm trong phần dữ liệu được cấp.

\- Loại cửa sổ vượt ranh giới.

\- Khoảng cách chống chồng lấn nhãn dùng horizon của khung: \*\*24 nến 1h hoặc 6 nến 4h\*\*, đều tương đương 24 giờ.

\- Đầu vào được chứa quá khứ trước ranh giới đánh giá vì đó là thông tin đã biết.

\- Không dùng dữ liệu kiểm chứng để chọn cấu hình, dừng sớm hoặc hiệu chỉnh đầu ra.



Thiếu dữ liệu sạch thì vẫn cho dự đoán và huấn luyện thử, nhưng khóa công nhận LoRA thắng và báo rõ phần còn thiếu.



\### 3.4. Công thức điểm



Ưu tiên gần hạn theo thời gian thực:



| Khoảng nhìn trước | Trọng số | Khung 1h | Khung 4h |

|---|---:|---|---|

| Đến 6 giờ | 50% | Bước 1–6 | Bước 1 |

| Trên 6 đến 12 giờ | 30% | Bước 7–12 | Bước 2–3 |

| Trên 12 đến 24 giờ | 20% | Bước 13–24 | Bước 4–6 |



Các bước trong cùng nhóm chia đều trọng số.



Chuẩn đối chiếu của mỗi khung là \*\*TimesFM base, bộ A, context 256, horizon tương ứng\*\*. Báo cáo thêm chuẩn “giá tương lai bằng giá hiện tại”.



Với mỗi đoạn `f`:



```text

A\_f = MAE có trọng số của mô hình / MAE có trọng số của base chuẩn



Q\_f = pinball loss có trọng số của mô hình

&#x20;     / pinball loss có trọng số của base chuẩn



L\_f = 0,70 × A\_f + 0,30 × Q\_f



Điểm = 100 × \[1 − (0,80 × trung bình L\_f + 0,20 × L\_f tệ nhất)]

```



\- Base chuẩn có điểm 0; điểm dương là tốt hơn base.

\- Điểm không phải xác suất dự đoán đúng.

\- Thành phần đoạn tệ nhất hạn chế thành tích chỉ tốt ở một giai đoạn.

\- Sai số chuẩn nhỏ hơn bước giá thì đánh dấu đoạn thiếu thông tin, không chia cho số gần 0.

\- Hai khung có bảng điểm riêng.

\- Ghim `score\_version=1`; thay công thức tạo bảng điểm mới.



Báo cáo bổ sung:



\- MAE theo USDT và RMSE.

\- Khung 1h: sai số tại bước 1, 6, 12, 24.

\- Khung 4h: sai số tại bước 1, 2, 3, 6.

\- Độ bao phủ và độ rộng dải 80%.

\- Độ đúng hướng; biến động dưới hai bước giá được ghi riêng.

\- Ngày thường/cuối tuần và nhóm biến động thấp/cao.

\- Số cửa sổ, mức chồng lấn và số khối độc lập.

\- So sánh với base và chuẩn giữ nguyên giá.



\### 3.5. Công nhận LoRA thắng



Ba trạng thái:



\- \*\*Ứng viên:\*\* học thành công.

\- \*\*Đứng đầu đợt thử:\*\* tốt nhất trên dữ liệu chọn cấu hình.

\- \*\*Đã kiểm chứng:\*\* vượt bài kiểm tra chưa được dùng để chọn nó.



Quy trình:



1\. Chọn tối đa ba cấu hình tốt trên ba đoạn đánh giá.

2\. Chạy lại với seed 42, 123, 2026; chọn theo điểm trung vị.

3\. Huấn luyện bản cuối từ base trên phần trước kiểm chứng; số vòng lấy từ kết quả dừng sớm trước đó.

4\. Khóa adapter, cấu hình và mã băm.

5\. Đánh giá đúng một ứng viên cuối trên phần kiểm chứng.

6\. Nếu đạt, sử dụng chính tệp đã kiểm chứng. Học thêm tạo ứng viên mới.



Điều kiện tự thay bản đang khuyến nghị:



\- Điểm cao hơn bản hiện tại ít nhất 2 điểm.

\- MAE tốt hơn base chuẩn và chuẩn giữ nguyên giá ít nhất 1%.

\- Không đoạn kiểm chứng nào có MAE tệ hơn base quá 5%.

\- Dải 80% có độ bao phủ 65–95%, đồng thời đáp ứng tiêu chí pinball trong điểm tổng.

\- Có ít nhất 20 khối dự đoán không chồng lấn.

\- Bootstrap theo khối cho khoảng tin cậy 95% của cải thiện MAE so với bản hiện tại nằm về phía tốt hơn.

\- Adapter đã lưu, nạp thử và sao lưu thành công.



\*\*Mỗi khối dài 24 giờ:\*\* 24 nến ở 1h hoặc 6 nến ở 4h. Vì vậy kiểm chứng mới cần tối thiểu khoảng \*\*20 ngày cho cả hai khung\*\*, có thể lâu hơn nếu thiếu dữ liệu.



Chia phần kiểm chứng thành ba đoạn liên tiếp để kiểm tra ổn định. Không đạt thì giữ bản hiện tại hoặc base; ứng viên vẫn được chọn thủ công với nhãn chưa kiểm chứng.



\### 3.6. Tự động tối ưu thiết lập



Dùng Optuna TPE để ưu tiên vùng cấu hình từng có kết quả tốt. Không xây hệ thống tự sửa mã. \[Cơ chế TPE](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html).



Mỗi đợt:



\- Cố định snapshot, cách chia đoạn, thước đo và không gian tìm kiếm.

\- Tối đa 30 lượt; 10 lượt đầu khám phá.

\- Tìm context, rank, learning rate, dropout, lịch sử học, bộ đầu vào và nhóm lớp LoRA.

\- Horizon cố định theo khung, không đưa vào tìm kiếm.

\- Alpha bằng 2 × rank, batch hiệu dụng 16, tối đa 10 vòng học.

\- Sau giai đoạn khám phá, kết thúc sớm nếu 12 lượt liên tiếp không cải thiện ít nhất 0,5 điểm.

\- Cấu hình tốt từ đợt trước được đề xuất lại nhưng phải chấm lại.

\- Không trộn điểm khác snapshot.

\- Lỗi tài nguyên được ghi riêng, không nhận điểm dự đoán giả.



Sau một đợt:



\- Có ứng viên đủ tốt: khóa và kiểm chứng.

\- Không có: giữ tự động bật, chờ ít nhất 7 ngày dữ liệu mới.

\- Phần kiểm chứng đã mở không được tái sử dụng làm bài thi mới.

\- Ứng viên tiếp theo phải khóa trước khoảng dữ liệu mới dùng kiểm chứng.

\- Mỗi khung chỉ có một ứng viên chờ kiểm chứng.



Không đảm bảo chất lượng tăng mãi. Khi chưa có bằng chứng tiến bộ, ứng dụng phải giữ bản đang dùng.



\## 4. Vận hành và năm thẻ web



\### 4.1. Hàng đợi và tài nguyên



\- Tối đa một tiến trình GPU.

\- Ưu tiên dự đoán → công việc thủ công → tự động.

\- Hai khung tự động luân phiên sau từng công việc nhỏ, chẳng hạn một fold.

\- Công việc huấn luyện tự động giới hạn 20 phút; hết thời gian không được coi là kết quả hoàn chỉnh.

\- Dự đoán chờ công việc hiện tại kết thúc, không ngắt giữa bước cập nhật.

\- Chống gửi trùng khi nhấn nhiều lần hoặc tải lại trang.



Mặc định context tối đa 512, batch GPU 2, `num\_workers=0` trên Windows. Giữ khoảng 4 GB RAM hệ thống còn trống; mục tiêu GPU dưới khoảng 12 GB.



Không giữ mô hình trong phiên Streamlit, không tích lũy tensor có gradient trong backtest, không bật `torch.compile` ở bản đầu.



Nếu hết bộ nhớ:



1\. Kết thúc tiến trình lỗi.

2\. Thử lại một lần với batch giảm một nửa, tăng tích lũy tương ứng.

3\. Ghi cấu hình thực tế.

4\. Không tự đổi context, dữ liệu hoặc mô hình.

5\. Vẫn lỗi thì chuyển chờ xử lý và giữ ý định tự động đang bật.



\### 4.2. Dừng và phục hồi



Trạng thái công việc:



`QUEUED → RUNNING → SUCCEEDED / FAILED / CANCELLED / INTERRUPTED`



Trạng thái tự động:



`SEARCHING`, `VALIDATING`, `WAITING\_DATA`, `WAITING\_AUDIT`, `PAUSED\_ERROR`, `STOPPED`.



\- Dừng được lưu vào SQLite, worker kiểm tra sau mỗi bước cập nhật và lô backtest.

\- Hủy công việc tự động chưa chạy, giữ checkpoint hợp lệ đã có.

\- Không công nhận lượt đang dở là thắng.

\- Heartbeat phát hiện treo hoặc tiến trình chết.

\- Chỉ thu hồi tiến trình ứng dụng tạo, xác minh PID và thời điểm khởi tạo.

\- Mở lại ứng dụng sẽ đối chiếu checkpoint và phục hồi, không chạy lại phần đã hoàn tất.

\- Đóng trình duyệt không dừng công việc.

\- Sau khi Windows khởi động lại, Khầy mở ứng dụng để tiếp tục; chưa làm dịch vụ tự khởi động.



\### 4.3. Bảo vệ LoRA thắng



Mỗi adapter có ID và thư mục bất biến, gồm:



\- `safetensors`, cấu hình LoRA và danh sách lớp.

\- Base revision, phiên bản mã và thư viện.

\- Timeframe, \*\*horizon\*\*, context, bộ đặc trưng và thứ tự cột.

\- Phạm vi học, snapshot, cách chia đoạn và seed.

\- Báo cáo kiểm chứng, trạng thái và SHA-256.



Lưu theo thứ tự:



1\. Ghi thư mục tạm cùng ổ.

2\. Hoàn tất ghi, đóng tệp và kiểm tra mã băm.

3\. Nạp thử, chạy dự đoán.

4\. Đổi tên thành thư mục chính thức.

5\. Đăng ký sẵn sàng trong SQLite.

6\. Xác minh bản sao trước khi đổi adapter đang khuyến nghị.



Không ghi đè `winner.safetensors`; bản thắng là tham chiếu đến ID đã xác minh.



\- Tự ghim mọi adapter từng được công nhận thắng.

\- Không tự xóa bản thắng, bản đang dùng hoặc dữ liệu cần tái lập chúng.

\- Xóa ứng viên thường qua thùng rác 7 ngày.

\- Gần đầy ổ thì nghỉ tự động.

\- SQLite được sao lưu bằng API backup.

\- Nhập adapter chỉ nhận dữ liệu và JSON hợp lệ, không nhận mã Python hoặc pickle.

\- Kiểm tra đường dẫn, dung lượng giải nén, mã băm và tính tương thích.

\- Có khôi phục bản trước và xuất bản sao sang ổ khác.



\### 4.4. Giao diện



Phía trên có bộ chọn 1h/4h, tình trạng dữ liệu, horizon tương ứng và hàng đợi GPU.



| Thẻ | Chức năng |

|---|---|

| \*\*Dự đoán\*\* | Chọn base/LoRA; nến lịch sử, trung vị và dải 80%; bảng \*\*24 bước ở 1h hoặc 6 bước ở 4h\*\*; lịch sử dự đoán |

| \*\*Backtest\*\* | Chọn mô hình và thời gian; nhãn ngoài mẫu/trùng dữ liệu học; biểu đồ sai số; so sánh; xuất CSV/JSON |

| \*\*Huấn luyện LoRA\*\* | Thiết lập thủ công; kiểm tra dữ liệu/tài nguyên; Bắt đầu/Dừng; tiến độ, loss, checkpoint |

| \*\*Tự động tối ưu\*\* | Bắt đầu/Dừng; lượt hiện tại; bảng điểm; lý do chờ và lý do công nhận/từ chối |

| \*\*Quản lý LoRA\*\* | Lọc theo khung; chi tiết; đổi tên; ghim; chọn dùng; xuất/nhập; sao lưu; lưu trữ; khôi phục |



Tiếng Việt, mặc định màu tối, điều khiển có nhãn rõ. Biểu đồ có phóng to/thu nhỏ và bảng số thay thế. Không diễn giải khoảng dự đoán thành độ chắc chắn của lệnh mua/bán.



\## 5. Giai đoạn triển khai và lệnh `/goal`



\### Quy ước bàn giao



P0 lưu:



\- `docs/paxg-lab/PLAN.md`: bản kế hoạch cập nhật này.

\- `docs/paxg-lab/PROGRESS.md`: tiến độ, quyết định, phần còn thiếu.

\- `docs/paxg-lab/phases/P0.md` đến `P7.md`: nhiệm vụ, nghiệm thu, bằng chứng và lệnh tiếp tục.



Mỗi giai đoạn phải đọc kết quả trước, giữ thay đổi không liên quan, kiểm thử đúng phạm vi và ghi bằng chứng thật. Báo tệp thay đổi, lệnh kiểm tra, kết quả, hạn chế và trạng thái Git. Không tự chạy sang giai đoạn kế tiếp.



Nếu dùng subagent, phân quyền sở hữu theo phần dữ liệu, mô hình, kiểm thử hoặc giao diện. Không cùng sửa một phần và không chạy hai công việc GPU.



\### P0 — Tài liệu, môi trường và chứng minh LoRA 3.0



\*\*Công việc\*\*



\- Lưu đặc tả cập nhật, tạo nhánh `codex/paxgusdt-lab`.

\- Tạo môi trường Python 3.12 riêng và khóa phiên bản.

\- Điểm xuất phát PyTorch 2.12.1 CUDA 13.0 cho Windows; kiểm tra thực tế RTX 5060 Ti.

\- Ghim checkpoint revision `43046b85ec22d584a13f8098c2ed39c889e129c2`.

\- Bổ sung preflight 3.0, không dùng ước lượng 2.5.

\- Xây đường tính toán dùng chung cho học và dự đoán.

\- Chạy base, cập nhật LoRA nhỏ, lưu/nạp lại.

\- Đo context 128/256/512, batch 1/2, horizon 24 và 6.



\*\*Nghiệm thu\*\*



CUDA chạy thật; gradient hợp lệ; base không đổi; adapter lưu/nạp đúng; đầu ra đúng hai horizon; kiểm thử suy luận cũ vẫn qua; có cấu hình phù hợp tài nguyên.



Không chứng minh được LoRA 3.0 thì ghi điểm chặn, không tự đổi xuống 2.5.



```text

/goal Thực hiện P0 của kế hoạch PAXG Forecast Lab cập nhật trong hội thoại: 1h dự đoán 24 nến, 4h dự đoán 6 nến. Lưu đầy đủ tài liệu vào docs/paxg-lab, dựng môi trường riêng và chứng minh TimesFM 3.0 huấn luyện LoRA, lưu/nạp lại được trên RTX 5060 Ti. Giữ thay đổi không liên quan, ghi bằng chứng và chỉ hoàn thành P0.

```



\### P1 — Dữ liệu và đặc trưng không rò rỉ



\*\*Công việc\*\*



\- Tải lịch sử/cập nhật tăng dần 1h và 4h.

\- Lưu nến, funding, mark price, báo cáo chất lượng.

\- Xây A/B/C, lịch UTC, snapshot và phân chia thời gian.

\- Chốt `DatasetSnapshot`, `FeatureSpec`, `SplitSpec`.

\- Tạo nhãn và khoảng loại cửa sổ theo horizon của từng khung.



\*\*Nghiệm thu\*\*



Không trùng bản ghi; không lọt nến đang chạy; phát hiện khoảng thiếu/sai OHLC; dữ liệu tương lai không đổi đặc trưng quá khứ; funding đúng thời điểm; kiểm tra chéo mẫu nến 4h với 1h.



```text

/goal Đọc docs/paxg-lab/PLAN.md và PROGRESS.md, thực hiện P1: kho dữ liệu Binance PAXGUSDT Futures 1h/4h, đặc trưng A/B/C, chất lượng dữ liệu, snapshot và phân chia không rò rỉ. Kiểm tra horizon 1h=24, 4h=6 xuyên suốt nhãn và ranh giới dữ liệu. Chỉ hoàn thành P1.

```



\### P2 — Base và backtest chuẩn



\*\*Công việc\*\*



\- Dự đoán trung vị và 9 phân vị theo horizon.

\- Backtest từng thời điểm, đọc theo lô.

\- Cài Score v1 với trọng số riêng theo khung.

\- Tạo chuẩn base và giữ nguyên giá.

\- Chốt `ForecastRequest`, `ForecastResult`, `BacktestSpec`, `ScoreReport`.



\*\*Nghiệm thu\*\*



Mốc thời gian đúng; cả hai khung kết thúc sau 24 giờ; đầu ra 24/6 bước; phân vị đúng; kiểm thử dự đoán hoàn hảo, giữ nguyên và cố tình kém; tái lập được báo cáo base thật.



```text

/goal Đọc bộ tài liệu docs/paxg-lab và thực hiện P2: dự đoán base TimesFM 3.0, 24 nến cho 1h và 6 nến cho 4h, cùng nhìn trước 24 giờ. Xây backtest và Score v1, kiểm tra thời gian, phân vị, trọng số và chống nhìn trước. Lưu báo cáo base cho cả hai khung; chỉ hoàn thành P2.

```



\### P3 — Huấn luyện thủ công và kho adapter



\*\*Công việc\*\*



\- Trainer, loss, dừng sớm và checkpoint.

\- Toàn bộ thiết lập thủ công.

\- Manifest, lưu nguyên tử, nạp kiểm tra, sao lưu.

\- Kiểm tra timeframe, horizon, đặc trưng và revision.

\- Chốt `TrainSpec`, `AdapterManifest`; backtest bằng P2.



\*\*Nghiệm thu\*\*



Chỉ LoRA thay đổi; chạy học thật cả hai khung; ngắt giữa lúc lưu không phá bản cũ; từ chối sai mã băm/sai horizon/thiếu tệp; chuỗi base→A→B→base không sót adapter; nhận diện dữ liệu học bị trùng trong backtest.



```text

/goal Đọc bộ tài liệu docs/paxg-lab và thực hiện P3: huấn luyện LoRA thủ công, checkpoint tốt nhất, manifest và lưu/nạp an toàn. Khóa đúng cặp 1h/24 và 4h/6, kiểm tra tương thích khi dùng adapter. Chạy thử thật cả hai khung và kiểm thử tiến trình chết khi lưu; chỉ hoàn thành P3.

```



\### P4 — Hàng đợi GPU và phục hồi



\*\*Công việc\*\*



\- Bộ điều phối và tiến trình con.

\- SQLite job, chống gửi trùng/chạy hai bộ điều phối.

\- Ưu tiên dự đoán, luân phiên hai khung, giới hạn thời gian.

\- Heartbeat, Dừng, phục hồi và quản lý tài nguyên.

\- Chốt `JobSpec`, `JobStatus`, `AutoRunState`.



\*\*Nghiệm thu\*\*



Tối đa một công việc GPU; dự đoán được ưu tiên sau lượt hiện tại; Dừng không sinh lượt mới; crash/restart không chạy đôi; bộ nhớ được thu hồi; không dừng tiến trình ngoài ứng dụng.



```text

/goal Đọc bộ tài liệu docs/paxg-lab và thực hiện P4: hàng đợi GPU một tiến trình, ưu tiên dự đoán, hai khung độc lập, dừng, heartbeat, giới hạn tài nguyên và phục hồi. Kiểm thử lỗi có kiểm soát, ghi bằng chứng và chỉ hoàn thành P4.

```



\### P5 — Web đủ năm thẻ



\*\*Công việc\*\*



\- Streamlit tiếng Việt và năm thẻ đã chốt.

\- Form chỉ gửi công việc khi nhấn nút.

\- Nến lịch sử, đường dự đoán, dải 80%, bảng bước theo khung.

\- Chọn base/LoRA, xem backtest và quản lý adapter.

\- Thẻ tự động chỉ bật chạy thật sau P6.

\- Lệnh mở ứng dụng một bước, bind `127.0.0.1`.



\*\*Nghiệm thu\*\*



Hoàn thành luồng dữ liệu→học→backtest→chọn LoRA→dự đoán trên web. Đổi thẻ/khung, tải lại hoặc mở hai tab không mất/trùng công việc. Khung 4h hiển thị đúng 6 giá tương lai. Có kiểm tra trình duyệt và ảnh bằng chứng.



```text

/goal Đọc bộ tài liệu docs/paxg-lab và thực hiện P5: web Streamlit tiếng Việt đủ năm thẻ, hai chế độ 1h/24 nến và 4h/6 nến, biểu đồ và chọn base/LoRA. Kết nối hàng đợi có sẵn, kiểm tra toàn bộ luồng trên trình duyệt và tạo lệnh mở một bước. Chỉ hoàn thành P5.

```



\### P6 — Tự tối ưu và công nhận LoRA thắng



\*\*Công việc\*\*



\- Optuna TPE có giới hạn từng đợt.

\- Dùng lại trainer, backtest và kho adapter.

\- Tìm→kiểm tra nhiều seed→khóa ứng viên→kiểm chứng→chọn thắng.

\- Khóa dữ liệu kiểm chứng khỏi tìm kiếm và dừng sớm.

\- Chờ dữ liệu, chờ kiểm chứng, Dừng và phục hồi.

\- Tính khối kiểm chứng theo 24 giờ ở cả hai khung.



\*\*Nghiệm thu\*\*



Bắt đầu/Dừng một nút; khởi động lại nhớ tiến độ; nhãn kiểm chứng không ảnh hưởng cấu hình được đề xuất trước khi mở; ứng viên kiểm chứng kém không thay bản đang dùng; mọi lần chọn thắng có tệp, bản sao và báo cáo.



Chạy một đợt nhỏ thật đủ chu trình. Chu kỳ dài được thử bằng dữ liệu/thời gian mô phỏng và phải ghi rõ.



```text

/goal Đọc bộ tài liệu docs/paxg-lab và thực hiện P6: tự tối ưu bằng Optuna TPE, backtest nhiều đoạn, kiểm chứng kín, công nhận LoRA thắng và chờ dữ liệu mới. Dùng horizon theo khung và tối thiểu 20 khối độc lập dài 24 giờ. Kiểm thử rò rỉ, phục hồi và chạy một đợt thật có giới hạn; chỉ hoàn thành P6.

```



\### P7 — Chạy dài và bàn giao



\*\*Công việc\*\*



\- Chạy liên tục tối thiểu 6 giờ, có công việc cả hai khung.

\- Ít nhất 20 chu kỳ tạo/kết thúc tiến trình GPU.

\- Thử mất mạng, API giới hạn, hết bộ nhớ, thiếu dung lượng, worker chết, khởi động lại.

\- Khôi phục adapter và SQLite từ bản sao.

\- Hoàn thiện hướng dẫn tiếng Việt, cách đọc điểm và xử lý sự cố.



\*\*Nghiệm thu\*\*



Không tăng bộ nhớ liên tục sau khi worker kết thúc; không vượt phần dự phòng; Dừng hoạt động; không còn tiến trình GPU mồ côi; LoRA thắng vẫn dùng được sau bài thử lỗi; bản sao khôi phục đúng.



Báo cáo cuối phân biệt kiểm tra thật, mô phỏng và phần chưa đủ dữ liệu thị trường. Không có chức năng đặt lệnh hoặc yêu cầu khóa giao dịch.



```text

/goal Đọc bộ tài liệu docs/paxg-lab và thực hiện P7: kiểm tra chạy dài, rò rỉ bộ nhớ, lỗi mạng, dừng/phục hồi, bảo toàn LoRA thắng và khôi phục bản sao. Chạy soak test tối thiểu 6 giờ, sửa lỗi trong phạm vi, hoàn thiện hướng dẫn và báo cáo nghiệm thu cuối.

```



\*\*Thứ tự:\*\* P0 → P1 → P2 → P3 → P4 → P5 → P6 → P7.



\*\*Mốc sử dụng:\*\* P5 có web dự đoán, huấn luyện và backtest thủ công; P6 có tự tối ưu; P7 nghiệm thu khả năng chạy lâu và phục hồi.




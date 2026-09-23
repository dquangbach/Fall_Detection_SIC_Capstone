# Fall Detection — CNN-LSTM & Leave-One-Subject-Out

Dự án phát hiện té ngã từ gia tốc ba trục, sử dụng **CNN-LSTM** và đánh giá khả năng tổng quát hóa sang người chưa xuất hiện trong tập train bằng **Leave-One-Subject-Out (LOSO)** trên **32 subject K-Fall**.

Pipeline chính nằm trong [model_cnn_lstm_loso.ipynb](model_cnn_lstm_loso.ipynb): đọc nhãn K-Fall, tạo cửa sổ, huấn luyện và đánh giá 32 fold, sau đó train model cuối trên dữ liệu của cả 32 subject. [model_cnn_lstm.ipynb](model_cnn_lstm.ipynb) là thử nghiệm đối chứng với split cố định và **không augmentation**; không phải nguồn kết quả LOSO.

Repository còn có thiết kế PCB, phần firmware ESP32-S3/MPU6050 và receiver Polar H10 phục vụ thử nghiệm. Model CNN-LSTM Keras hiện tại và model nhúng trong firmware là các bộ artifact khác nhau.

## Cấu trúc dự án

```text
.
├── model_cnn_lstm_loso.ipynb     # Pipeline chính: LOSO + train model cuối
├── model_cnn_lstm.ipynb          # Đối chứng: split subject cố định, no augmentation
├── saved_models_cnn_lstm/
│   ├── best_model_SAxx.keras    # Checkpoint của fold có SAxx làm test
│   ├── final_selection_best.keras
│   ├── final_model.keras       # Model train lại trên toàn bộ 32 subject
│   ├── scaler.pkl              # Scaler đi cùng final_model.keras
│   ├── metadata.pkl            # Features, window, threshold, thông tin train cuối
│   ├── best_model_noaug.keras  # Checkpoint thử nghiệm no augmentation
│   └── best_model.keras        # Checkpoint thử nghiệm trước đó
├── kfall/
│   ├── label_data_new/         # Nhãn onset/impact: SAxx_label.xlsx
│   └── sensor_data_new/        # CSV theo subject, chuẩn bị riêng ở máy local
├── polar_h10_receiver/         # Thử nghiệm inference Python từ UDP Polar H10
├── esp32s3-tinyml-fall-detector/ # Repository firmware được ghi nhận bằng gitlink
├── fall_detection_pcb/         # Schematic, PCB, footprint KiCad và PDF
├── fall_detection_dataset/     # Dữ liệu local của pipeline cũ
└── doc/                        # Tài liệu tham khảo local
```

Các notebook cũ như `model.ipynb` và `model_3_features.ipynb` không còn trong cây dự án hiện tại. Bắt đầu với **`model_cnn_lstm_loso.ipynb`**.

## Dữ liệu và nhãn

Notebook chính đọc trực tiếp:

```text
kfall/
├── label_data_new/
│   ├── SA06_label.xlsx
│   └── ...
└── sensor_data_new/
    ├── SA06/
    │   ├── S06T01R01.csv
    │   └── ...
    └── ...
```

- Danh sách subject được lấy từ các thư mục bắt đầu bằng `SA` trong `sensor_data_new`.
- Tên CSV theo mẫu `SxxTxxRxx.csv`, chứa subject, task và trial.
- CSV cần các cột `FrameCounter`, `AccX`, `AccY`, `AccZ`. Dữ liệu được dùng theo thứ tự dòng có sẵn trong file; cần giữ thứ tự thời gian.
- File Excel nhãn sử dụng `Task Code (Task ID)`, `Trial ID`, `Fall_onset_frame`, `Fall_impact_frame`.
- Nhãn mẫu là ngã nếu `onset <= FrameCounter <= impact`. Recording không có trong bảng nhãn được xem là bình thường; cần chuẩn bị đầy đủ các file nhãn để tránh gán sai.
- Pipeline này không cần cột `FallCheck` và không đọc từ `Sample_Training/Sample_Test` của pipeline cũ.

Dữ liệu gia tốc dùng đơn vị **g**, với bước thời gian 0.01 giây (100 Hz) trong CSV K-Fall hiện có. Notebook không tự resample hoặc chuyển đơn vị.

### Cửa sổ và chuẩn hóa

| Thiết lập | Giá trị |
|---|---|
| Features theo thứ tự | `AccX, AccY, AccZ` |
| Window size | 200 mẫu, khoảng 2 giây ở 100 Hz |
| Window step | 100 mẫu, overlap 50% |
| Nhãn cửa sổ | Fall nếu ít nhất 20% mẫu nằm trong khoảng onset–impact |
| Input model | `(batch, 200, 3)`, không flatten |
| Output | Một sigmoid score cho lớp Fall |
| Decision threshold | `score >= 0.50` → Fall |

Cửa sổ được tạo riêng trong từng recording; không nối hai recording hoặc hai subject. Đoạn cuối thiếu 200 mẫu bị bỏ qua. Cửa sổ từng subject được cache trong RAM để tái sử dụng qua các fold.

Trong mỗi fold, `StandardScaler` được fit **chỉ trên cửa sổ train gốc**, theo từng feature bằng `reshape(-1, 3)`. Scaler đó transform train, train augmentation, validation và test. Không fit scaler trên subject test.

### Augmentation của pipeline chính

Chỉ áp dụng lên tập train, trước chuẩn hóa:

| Phép biến đổi | Xác suất trên mỗi bản sao cửa sổ | Cấu hình |
|---|---:|---|
| Jitter | 0.10 | Nhiễu Gaussian, mean 0, sigma 0.05 |
| Scaling | 0.10 | Một hệ số chung cho Acc XYZ, Gaussian mean 1, sigma 0.10 |
| Rotation | 0.10 | Góc ngẫu nhiên ±10° mỗi trục, cùng phép quay cho toàn bộ cửa sổ |

Ba phép được chọn độc lập nên một cửa sổ có thể nhận nhiều phép hoặc không nhận phép nào. Ghép tập gốc với tập bản sao để tăng gấp đôi số window train; nhãn giữ nguyên. Validation/test của LOSO không augmentation.

## Kiến trúc CNN-LSTM

```text
Input (200, 3)
  → Conv1D(32, kernel=5, same) → BatchNorm → ReLU → MaxPool(2) → Dropout(0.2)
  → Conv1D(64, kernel=3, same) → BatchNorm → ReLU → MaxPool(2) → Dropout(0.2)
  → LSTM(64, dropout=0.2, recurrent_dropout=0)
  → Dense(32, ReLU) → Dropout(0.3)
  → Dense(1, sigmoid)
```

CNN trích đặc trưng cục bộ theo thời gian; LSTM xử lý chuỗi đặc trưng sau hai bước pooling. Loss là `binary_crossentropy`; optimizer Adam bắt đầu tại learning rate `1e-3`. Class weight được tính theo `N / (2 × N_class)` để cân bằng hai lớp.

## Đánh giá Leave-One-Subject-Out

Với 32 subject, notebook chạy 32 fold. Trong mỗi fold:

1. Giữ 1 subject làm **test độc lập**.
2. Chọn xoay vòng 2 subject trong 31 subject còn lại làm validation.
3. Dùng 29 subject còn lại để train, augmentation và fit scaler.
4. Khởi tạo CNN-LSTM mới, chọn checkpoint theo validation.
5. Nạp checkpoint tốt nhất và dự đoán subject test tại threshold cố định 0.50.

Mỗi subject làm test đúng một lần. Đây là LOSO có validation tách theo subject, không phải chia ngẫu nhiên window và cũng không phải nested LOSO để tìm hyperparameter.

### Cấu hình huấn luyện mỗi fold

| Thiết lập | Giá trị trong code |
|---|---|
| Batch size | 32 |
| Epoch tối đa | 50 |
| ModelCheckpoint | `val_recall`, `mode="max"`, lưu best |
| EarlyStopping | `val_recall`, patience 12, min_delta `1e-3`, restore best weights |
| ReduceLROnPlateau | `val_loss`, patience 3, factor 0.5, min_lr `1e-5` |
| Threshold test | 0.50, không quét ngưỡng trên test |

Checkpoint lưu thành `saved_models_cnn_lstm/best_model_<test_subject>.keras`. Test dùng checkpoint tải lại; `ModelCheckpoint` không đặt `min_delta` như `EarlyStopping`, nên hai callback có thể chọn epoch khác nhau.

### Báo cáo kết quả

Cell tổng hợp tạo:

- Bảng mỗi fold: số window, phân bố lớp, Accuracy, Precision/Recall/F1/F2 lớp Fall, ROC-AUC và TN/FP/FN/TP.
- Mean, standard deviation, min và max qua các fold. ROC-AUC là NaN nếu test fold chỉ có một lớp.
- Confusion matrix và classification report gộp dự đoán của toàn bộ subject test.
- Biểu đồ Precision và Recall theo subject.

Mean qua subject và metrics gộp theo window là hai cách tổng hợp khác nhau. Các chỉ số ở đây đánh giá **cửa sổ**, không tự tương đương số sự kiện ngã phát hiện được hoặc số cảnh báo giả mỗi giờ.

**Tình trạng kết quả trong repository:** có 32 checkpoint theo subject, nhưng notebook LOSO hiện lưu với output rỗng và chưa có bảng metrics LOSO được export. Vì vậy README không gán số đo từ notebook đối chứng hay validation của model cuối cho LOSO. Để công bố kết quả, lưu output cell tổng hợp và bảng từng fold sau lần chạy tương ứng.

Scalers từng fold và dự đoán test chưa được lưu thành artifact riêng. Cell mục 7 có thể nạp trực tiếp 32 file `best_model_SAxx.keras` và dựng lại scaler từ đúng 29 subject train gốc của mỗi fold, không cần train lại model. Việc này yêu cầu giữ nguyên dataset, nhãn, thứ tự subject, features, window và cách chia fold như khi train checkpoint. `scaler.pkl` hiện có thuộc model cuối; không dùng scaler này để đánh giá checkpoint LOSO.

## Train model cuối trên 32 subject

Cell cuối của notebook chính dùng `FINAL_USE_AUGMENTATION = True`, seed 42 và thực hiện hai pha:

1. **Chọn epoch:** ghép dữ liệu 32 subject, fit scaler trên toàn bộ window gốc, tạo augmentation, chuẩn hóa và shuffle. Giữ 10% window bằng `validation_split=0.10`; train tối đa 100 epoch, với checkpoint/early stopping theo `val_recall` (patience 12) và giảm LR theo `val_loss` (patience 3). Chọn `argmax(val_recall) + 1`.
2. **Train lại toàn bộ:** khởi tạo CNN-LSTM mới và train trên tất cả window gốc + augmentation của 32 subject đến epoch vừa chọn. Pha này không giữ validation, không dùng early stopping hoặc ReduceLROnPlateau; Adam bắt đầu lại tại `1e-3`. Lưu thành `final_model.keras`.

Validation pha 1 là split theo window sau khi augmentation và fit scaler toàn bộ; các window overlap hoặc bản gốc/bản sao có thể nằm hai phía của split. Vì vậy nó chỉ là tín hiệu chọn epoch nội bộ, **không phải đánh giá subject độc lập**. Lịch giảm LR ở pha 1 cũng không được phát lại trong pha train toàn bộ. Chất lượng cần được báo cáo bằng protocol đánh giá phù hợp, không bằng training score của model cuối.

File `metadata.pkl` hiện ghi: **32 subject, augmentation bật, selected_epoch = 13**, `selected_val_recall ≈ 0.996633`. Đây là thông tin pha chọn epoch của lần export, không phải Recall LOSO hay kết quả kiểm thử độc lập của `final_model.keras`.

### Bộ artifact để inference

| File trong `saved_models_cnn_lstm/` | Vai trò |
|---|---|
| `final_model.keras` | CNN-LSTM train lại trên toàn bộ dữ liệu |
| `scaler.pkl` | StandardScaler fit trên window gốc của 32 subject |
| `metadata.pkl` | Feature order, window size/step, threshold, subject và cấu hình train cuối |
| `final_selection_best.keras` | Checkpoint pha chọn epoch; không phải model train lại toàn bộ |
| `best_model_SAxx.keras` | Model fold LOSO có subject SAxx làm test |
| `best_model_noaug.keras` | Checkpoint thử nghiệm đối chứng no augmentation |

Luôn sử dụng `final_model.keras`, `scaler.pkl` và `metadata.pkl` từ cùng lần export. Khi inference, chỉ transform bằng scaler đã lưu; không fit lại scaler, không augmentation.

## Cài đặt và chạy

Từ thư mục gốc repository:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install tensorflow numpy pandas scikit-learn matplotlib joblib openpyxl jupyterlab
python -m jupyter lab
```

Trên Windows, kích hoạt bằng `.venv\Scripts\activate`. Chưa có file khóa dependency Python trong repository; cần lưu phiên bản môi trường cho các lần chạy cần tái lập.

1. Chuẩn bị đầy đủ CSV và Excel nhãn theo cấu trúc `kfall/` ở trên.
2. Mở `model_cnn_lstm_loso.ipynb`, chọn kernel đúng môi trường, chạy với working directory là gốc repository.
3. Chạy các cell import, đọc nhãn, cấu hình, cache dữ liệu, augmentation và định nghĩa model.
4. Nếu đã có 32 checkpoint, chạy thẳng **mục 7** để nạp model và đánh giá; chỉ cần import và các mục 1–3, bỏ qua mục 6. Nếu cần train lại, chạy mục 6 trước. Lưu notebook cùng output khi cần báo cáo.
5. Sau khi chốt cách đánh giá, chạy cell cuối để tạo model cho toàn bộ 32 subject.

Nếu chỉ cần train model cuối sau khi đã hoàn tất LOSO, có thể chạy các cell chuẩn bị ở bước 3 rồi chuyển thẳng đến cell cuối; không bắt buộc chạy lại 32 fold. Chạy lại sẽ ghi đè các artifact cùng tên trong `saved_models_cnn_lstm/`; sao lưu bộ cần giữ trước khi thực hiện.

LOSO hiện không đặt seed riêng cho từng fold, trong khi cell final đặt seed 42. Các lần chạy LOSO có thể khác nhau do khởi tạo trọng số, dropout và augmentation ngẫu nhiên.

### Notebook đối chứng

`model_cnn_lstm.ipynb` giữ 6 subject test cố định:
`SA21, SA23, SA24, SA26, SA30, SA33`; chia 26 subject còn lại thành 21 train và 5 validation. Cấu hình hiện tại không augmentation, tối đa 100 epoch, early stopping patience 12 và LR patience 3. Kết quả của notebook này phục vụ đối chiếu, không thay thế đánh giá LOSO.

## Thử nghiệm trực tiếp với Polar H10

Khi có thư mục `polar_h10_receiver/`, chạy trong môi trường đã cài dependencies:

```bash
python polar_h10_receiver/udp_acc_receiver.py --check-model
python polar_h10_receiver/udp_acc_receiver.py
```

Sender gửi UDP đến IP máy tính, port **5005**, theo `timestamp,x_mg,y_mg,z_mg`, với tốc độ đã cấu hình **100 mẫu/giây**. Receiver:

- Đổi mg → g và đổi trục thành `(Polar Y, -Polar X, -Polar Z)`.
- Gom window 200 mẫu, dự đoán lại mỗi 100 mẫu, dùng bộ artifact final.
- In sigmoid score và nhãn tại threshold trong metadata.
- Bỏ mẫu lỗi/trùng/đảo timestamp; reset sau 2 giây không nhận mẫu hợp lệ.

Mặc định đếm frame ở 100 Hz, không resample. Khi biết đơn vị timestamp, có thể thêm `--timestamp-unit ns` (hoặc `us/ms`) để kiểm tra nhịp lấy mẫu và khoảng mất mẫu. Thêm `--show-acc` để xem gia tốc.

Mapping trục giữ từ thử nghiệm receiver; hướng gắn cảm biến, dấu trục Z và khác biệt vị trí đo so với K-Fall cần được xác nhận thực nghiệm. Kết quả LOSO trên K-Fall không tự đại diện cho chất lượng nhận diện trực tiếp trên Polar.

## Phần cứng và trạng thái triển khai

- `fall_detection_pcb/`: thiết kế KiCad, schematic, PCB, footprint ESP32-S3/header và các PDF lớp mạch.
- `esp32s3-tinyml-fall-detector/`: phần firmware ESP32-S3 + MPU6050, lấy mẫu, preprocessing, TFLite Micro, LED và Wi-Fi/MQTT.

Firmware đang là repository riêng, được repository này tham chiếu bằng gitlink; hiện chưa có `.gitmodules` để tự khởi tạo bằng `git submodule update`. Khi clone mới, cần lấy firmware từ repository riêng và đối chiếu revision được ghi nhận.

Bản firmware local đang có thay đổi chưa commit cho input INT8 6 features `(200 × 6)`, step 50; revision firmware được repository chính tham chiếu vẫn dùng cấu hình 8 features. Cả hai đều khác bộ CNN-LSTM final với input float32 `(1,200,3)`. Chưa có artifact TFLite/C của CNN-LSTM final trong `saved_models_cnn_lstm/`; cần thực hiện export, đồng bộ preprocessing và kiểm tra runtime trước khi triển khai model này lên ESP32-S3.

## Dữ liệu và quản lý repository

CSV và file H5 bị loại theo `.gitignore`; thư mục sensor K-Fall cần chuẩn bị riêng khi clone. Các Excel nhãn K-Fall và bộ checkpoint Keras hiện được Git theo dõi. `doc/`, môi trường ảo và cache Python là dữ liệu local.

Thêm một thư mục vào `.gitignore` không tự bỏ tracking các file đã commit. Receiver Polar hiện vẫn có file được Git theo dõi; cần phân biệt tình trạng đó với quy tắc ignore local.

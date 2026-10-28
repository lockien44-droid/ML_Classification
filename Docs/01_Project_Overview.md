# 1. Tổng quan dự án và quy trình A–Z

## 1.1 Bài toán

Dự án nhận nội dung email và dự đoán email thuộc một trong hai lớp:

- `0`: không spam (ham).
- `1`: spam.

Đây là **phân loại nhị phân có giám sát**: dữ liệu huấn luyện gồm email đã có nhãn, mô hình học mối liên hệ giữa từ ngữ và nhãn.

## 1.2 Mục tiêu học tập

Sau khi hoàn thành, người học có thể giải thích:

- Cách đọc, kiểm tra và làm sạch dữ liệu văn bản.
- Vì sao cần chia Train/Test và giữ tập Test độc lập.
- Cách TF-IDF biến email thành vector số.
- Cách dùng Logistic Regression và Linear SVM để phân loại.
- Cách dùng 5-fold cross-validation để chọn siêu tham số.
- Cách đọc Accuracy, Precision, Recall, F1 và confusion matrix.
- Cách dùng pipeline đã lưu để dự đoán email mới.

## 1.3 Cấu trúc thành phần

| Thành phần | Vai trò |
|---|---|
| `notebooks/Part 1 - Data Preparation.ipynb` | Khám phá, làm sạch và chia dữ liệu. |
| `notebooks/Part 2 - Model Training and Tuning.ipynb` | Tạo TF-IDF pipeline, tìm siêu tham số bằng CV và lưu mô hình. |
| `notebooks/Part 3 - Evaluation and Interpretation.ipynb` | Đánh giá một lần trên Test, xem lỗi và lưu biểu đồ. |
| `src/preprocess.py` | Hàm `clean_text`, chuẩn bị dữ liệu, chia và lưu CSV. |
| `src/models.py` | Cấu hình Pipeline và lưới siêu tham số. |
| `src/evaluation.py` | Hàm tính chỉ số và lưu biểu đồ. |
| `Model/` | Factory Logistic Regression/SVM được `src.models` sử dụng. |
| `data/raw/` | CSV gốc; không chỉnh sửa trực tiếp. |
| `data/processed/` | CSV sạch, Train và Test do Part 1 tạo. |
| `artifacts/` | Pipeline đã fit, dùng lại cho Part 3 và dự đoán. |
| `results/` | Kết quả CV, metrics và biểu đồ. |
| `docs/` | Tài liệu hướng dẫn và kiến thức nền. |

## 1.4 Luồng A–Z

```text
CSV gốc
  → kiểm tra dữ liệu
  → làm sạch và loại trùng
  → chia Train/Test 80/20 có phân tầng
  → dùng Train để tìm pipeline tốt bằng Stratified 5-fold CV
  → fit lại pipeline tốt trên toàn bộ Train
  → lưu pipeline
  → đánh giá một lần trên Test
  → phân tích lỗi và lưu kết quả
  → dùng pipeline đã lưu để dự đoán email mới
```

### A. Chuẩn bị môi trường

Tạo môi trường ảo, cài `requirements.txt`, mở Jupyter và chọn kernel của môi trường đó. Hướng dẫn lệnh chi tiết nằm trong [README gốc](../README.md#cài-đặt).

### B. Chạy Part 1

Mở Part 1 và chạy tất cả cell từ trên xuống dưới. Xem số dòng, cột, giá trị thiếu, phân bố nhãn và trùng lặp. Sau làm sạch, notebook tạo:

- `data/processed/emails_clean.csv`
- `data/processed/train.csv`
- `data/processed/test.csv`

Kết quả hiện tại: 5.171 email ban đầu → 4.894 email sạch; Train 3.915, Test 979.

### C. Chạy Part 2

Mở Part 2. Notebook chỉ fit trên Train. GridSearchCV chạy 5 folds cho 12 cấu hình của từng mô hình, chọn theo F1 Spam. Test được đọc để xác nhận file/cột nhưng không dùng trong tìm kiếm.

Kết quả Part 2:

- `artifacts/lr_tuned_pipeline.joblib`
- `artifacts/svm_tuned_pipeline.joblib`
- `results/cv_tuning_results.csv`

### D. Chạy Part 3

Mở Part 3 sau khi Part 2 đã lưu artifacts. Notebook tải hai pipeline và tạo dự đoán trên Test. Nó lưu metrics, confusion matrix, biểu đồ so sánh và hiển thị các email mà mô hình bất đồng hoặc sai.

### E. Dự đoán email mới

Sau Part 2, có thể chạy `python main.py`, chọn `1`, hoặc chạy `python predict.py`. Chương trình dùng cùng `clean_text()` như Part 1 rồi tải cả hai pipeline tuned.

## 1.5 Quy tắc quan trọng

- Không fit TF-IDF trên toàn bộ dữ liệu trước khi chia hoặc trước cross-validation.
- Không dùng Test để chọn siêu tham số hay mô hình.
- Nếu dữ liệu hoặc cách làm sạch đổi, chạy lại Part 1 → Part 2 → Part 3.
- Pipeline tuned và artifacts của script cũ có tên khác nhau; xem [README gốc](../README.md#lưu-ý-về-đường-dẫn) trước khi dùng các script cũ.

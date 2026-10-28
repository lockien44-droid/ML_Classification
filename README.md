# Phân loại email spam bằng Logistic Regression và Linear SVM

Dự án phân loại email tiếng Anh thành hai lớp: `0` — không spam, `1` — spam. Quy trình chính được trình bày và chạy theo thứ tự trong ba Jupyter Notebook; các hàm dùng lại nằm trong `src/`.

## Cấu trúc dự án

```text
ML_Classification/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── Part 1 - Data Preparation.ipynb
│   ├── Part 2 - Model Training and Tuning.ipynb
│   └── Part 3 - Evaluation and Interpretation.ipynb
├── data/
│   ├── raw/
│   │   └── spam_ham_dataset.csv
│   └── processed/
│       ├── emails_clean.csv
│       ├── train.csv
│       └── test.csv
├── src/
│   ├── __init__.py
│   ├── preprocess.py
│   ├── models.py
│   └── evaluation.py
├── Model/              # Các module tạo classifier/feature cũ, vẫn được dùng bởi src.models
├── artifacts/          # Pipeline đã fit
├── results/            # Bảng metrics/CV
│   └── figures/        # Confusion matrix và biểu đồ so sánh
├── docs/               # Tài liệu dự án
└── deprecated/         # Mã/notebook cũ nếu được lưu lại
```

Notebook là luồng làm việc chính. `main.py` và `predict.py` là lối chạy CLI để dự đoán email mới. Một số script gốc như `train.py`, `eda.py`, `src/evaluate.py` còn giữ cho quy trình cũ; chúng không phải luồng Part 1–3 được mô tả bên dưới.

Tài liệu học tập chi tiết từ dữ liệu đến dự đoán nằm trong [`docs/README.md`](docs/README.md).

## Dữ liệu

Dữ liệu gốc là [Spam Mails Dataset trên Kaggle](https://www.kaggle.com/datasets/venky73/spam-mails-dataset), đặt tại `data/raw/spam_ham_dataset.csv`. Notebook sử dụng hai cột:

- `text`: nội dung email.
- `label_num`: nhãn số, `0` không spam và `1` spam.

Part 1 kiểm tra thiếu/trùng, giữ nội dung gốc trong `raw_text`, chuyển `text` về chữ thường, bỏ thẻ HTML, chuẩn hóa khoảng trắng và loại nội dung trùng trước/sau làm sạch. Với dữ liệu hiện có, 5.171 dòng đầu vào giảm còn 4.894 email: 3.432 không spam và 1.462 spam.

## Cài đặt

Mở PowerShell tại thư mục gốc dự án:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Mở Jupyter Notebook:

```powershell
python -m notebook
```

Chọn kernel Python của môi trường `.venv`, rồi chạy notebook theo thứ tự Part 1 → Part 2 → Part 3. Nếu cần đăng ký kernel để chọn rõ ràng:

```powershell
python -m ipykernel install --user --name ml-classification --display-name "Python (ML Classification)"
```

## Quy trình chạy notebook

### Part 1 — Data Preparation

Mở `notebooks/Part 1 - Data Preparation.ipynb` và chạy **Run All**. Notebook import các hàm dùng lại từ `src/preprocess.py`, đọc dữ liệu gốc, kiểm tra chất lượng, làm sạch, rồi chia có phân tầng:

- 80% Train, 20% Test.
- `random_state=42` để phép chia có thể tái lập.
- `stratify=label_num` để tỷ lệ hai lớp gần tương đương giữa hai tập.

Đầu ra trong `data/processed/`:

- `emails_clean.csv`: dữ liệu đã làm sạch.
- `train.csv`: dữ liệu huấn luyện.
- `test.csv`: dữ liệu giữ riêng để đánh giá cuối.

### Part 2 — Model Training and Tuning

Mở `notebooks/Part 2 - Model Training and Tuning.ipynb` sau Part 1. Notebook đọc Train và Test, nhưng **chỉ Train được dùng trong tìm kiếm tham số**. Test được giữ lại cho Part 3.

Notebook so sánh Logistic Regression và Linear SVM. Mỗi mô hình được ghép với `TfidfVectorizer` trong cùng `Pipeline`, nên vectorizer được fit riêng bên trong từng fold. `GridSearchCV` dùng Stratified 5-fold CV (`shuffle=True`, seed `42`) và tối ưu F1 của lớp spam (`label_num=1`). Lưới gồm:

- `ngram_range`: unigram `(1, 1)` hoặc unigram + bigram `(1, 2)`.
- `min_df`: `1` hoặc `2`.
- `C`: `0.1`, `1.0`, `10.0`.

Grid có 12 cấu hình cho mỗi mô hình. `refit=True` fit cấu hình tốt nhất trên toàn bộ Train sau CV. Chỉ sau khi cả hai lượt tìm kiếm thành công, notebook lưu:

- `artifacts/lr_tuned_pipeline.joblib`
- `artifacts/svm_tuned_pipeline.joblib`
- `results/cv_tuning_results.csv`

### Part 3 — Evaluation and Interpretation

Mở `notebooks/Part 3 - Evaluation and Interpretation.ipynb` sau Part 2. Mỗi pipeline được dự đoán trên Test; notebook báo cáo Accuracy, Precision, Recall, F1 của lớp spam, confusion matrix, các trường hợp hai mô hình bất đồng và ví dụ dự đoán sai.

Đầu ra:

- `results/metrics.csv`
- `results/figures/confusion_matrix_lr.png`
- `results/figures/confusion_matrix_svm.png`
- `results/figures/model_metrics_comparison.png`

Confusion matrix dùng hàng là nhãn thật và cột là nhãn dự đoán theo thứ tự `[không spam, spam]`. Điểm `decision_function` của Linear SVM là điểm quyết định, **không phải xác suất**. Logistic Regression có `predict_proba`; xác suất này không đồng nghĩa độ tin cậy đã được hiệu chỉnh.

## Dự đoán email mới

Sau khi Part 2 đã tạo hai tuned pipeline, chạy:

```powershell
python main.py
```

Chọn `1` và nhập nội dung email tiếng Anh. Có thể gọi thẳng:

```powershell
python predict.py
```

`predict.py` dùng chung `clean_text()` trong `src/preprocess.py`, rồi đưa văn bản đã làm sạch vào cả hai pipeline. Kết quả hiển thị nhãn của hai mô hình, xác suất spam của Logistic Regression và điểm quyết định (không phải xác suất) của SVM.

## Kết quả hiện tại

Kết quả đã lưu từ lần chạy hiện tại (5-fold CV trên Train, đánh giá một lần trên Test):

| Mô hình | F1 CV Spam | Accuracy Test | Precision Spam | Recall Spam | F1 Spam |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9805 | 0.9918 | 0.9897 | 0.9829 | 0.9863 |
| Linear SVM | 0.9821 | 0.9959 | 0.9932 | 0.9932 | 0.9932 |

Đây là kết quả trên một phép chia Train/Test cụ thể, không đảm bảo hiệu năng tương tự trên dữ liệu ngoài dự án.

## Dependencies

Các thư viện cần thiết được liệt kê trong `requirements.txt`:

- `pandas` — đọc và xử lý CSV.
- `scikit-learn` — TF-IDF, Pipeline, classifier, cross-validation và metrics.
- `joblib` — lưu/tải pipeline đã huấn luyện.
- `matplotlib` — tạo biểu đồ.
- `notebook`, `ipykernel` — chạy Jupyter Notebook.

## Lưu ý về đường dẫn

Mã nguồn và notebook dùng các đường dẫn chữ thường `data/` và `docs/`. Bản checkout hiện tại trên Windows hiển thị thư mục là `Data/` và `Docs/`; Windows không phân biệt hoa/thường nên đường dẫn vẫn hoạt động tại máy này. Trên hệ điều hành phân biệt hoa/thường (ví dụ Linux), cần chuẩn hóa tên thư mục trong Git thành `data/` và `docs/` để khớp với code.

Các script cũ `train.py`, `eda.py`, `src/evaluate.py` vẫn dùng một số artifacts/đầu ra cũ như `lr_pipeline.joblib`, `svm_pipeline.joblib`, `results/eda/` và `results/cv_results.csv`. Luồng notebook mới dùng `*_tuned_pipeline.joblib`, `results/cv_tuning_results.csv` và các output Part 3 nêu trên; không trộn hai bộ artifacts khi chạy.

## Giới hạn

- Dữ liệu và ví dụ dự đoán bằng tiếng Anh.
- Email ngắn, từ viết biến dạng hoặc nội dung quảng cáo hợp lệ có thể gây khó khăn cho mô hình.
- Nhãn không đánh giá độ an toàn của URL hoặc tệp đính kèm.

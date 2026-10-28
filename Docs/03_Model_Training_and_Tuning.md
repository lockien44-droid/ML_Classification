# 3. Huấn luyện và tìm siêu tham số

Notebook thực hành: [`Part 2 - Model Training and Tuning.ipynb`](../notebooks/Part%202%20-%20Model%20Training%20and%20Tuning.ipynb)

## 3.1 Vì sao cần chuyển văn bản thành số?

Các bộ phân loại scikit-learn nhận đầu vào số. `TfidfVectorizer` tách email thành token và tạo vector thưa: mỗi cột đại diện một từ/cụm từ, mỗi hàng đại diện một email.

TF-IDF kết hợp:

- **TF (Term Frequency):** mức hiện diện của token trong email.
- **IDF (Inverse Document Frequency):** giảm trọng số token quá phổ biến trên toàn bộ tài liệu.

Trong pipeline, vectorizer dùng `stop_words="english"`. Tham số `ngram_range` xác định token:

- `(1, 1)`: unigram — từng từ một.
- `(1, 2)`: unigram và bigram — cả từ đơn lẫn cặp từ liền kề.

## 3.2 Hai classifier

### Logistic Regression

Là thuật toán phân loại tuyến tính. Nó tính điểm tuyến tính từ đặc trưng và chuyển thành xác suất lớp. Trong dự án, `create_model()` nằm trong `Model/logistic_regression.py` và dùng `max_iter=1000`.

### Linear SVM

`LinearSVC` tìm ranh giới phân tách tuyến tính với biên lớn giữa lớp. Nó phù hợp với vector văn bản thưa, nhiều chiều. Trong dự án, cấu hình cơ sở nằm ở `Model/svm.py`: `C=1.0`, `max_iter=20000`, `random_state=42`.

## 3.3 Pipeline tránh rò rỉ

`src/models.py` tạo:

```python
Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("classifier", classifier),
])
```

Pipeline được truyền vào GridSearchCV. Mỗi fold chỉ fit TF-IDF trên phần huấn luyện của fold đó; văn bản ở fold validation chỉ được transform. Nếu fit vectorizer trên toàn bộ Train trước CV, vocabulary/IDF đã nhìn thấy validation folds và điểm CV có thể lạc quan.

## 3.4 5-fold cross-validation

`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` chia Train thành 5 fold, giữ gần đúng tỷ lệ spam ở mỗi fold. Mỗi vòng dùng 4 fold để fit và 1 fold để validation; mỗi fold lần lượt làm validation một lần.

Part 2 dùng `GridSearchCV` với:

```text
ngram_range: (1,1), (1,2)    2 lựa chọn
min_df:      1, 2            2 lựa chọn
C:           0.1, 1, 10      3 lựa chọn
Tổng:                         12 cấu hình mỗi classifier
```

12 cấu hình × 5 folds = 60 lượt CV fit cho mỗi mô hình. `refit=True` fit lại cấu hình tốt nhất trên toàn bộ Train sau khi tìm kiếm hoàn tất.

## 3.5 Tiêu chí chọn

`scoring="f1"` trong bài toán nhị phân tối ưu F1 của nhãn dương `1` — Spam. F1 cân bằng Precision và Recall. Nó hữu ích khi cả việc chặn nhầm email thường lẫn bỏ sót spam đều cần theo dõi.

Test được đọc để kiểm tra file và cột, nhưng không truyền vào `.fit()` hoặc GridSearchCV.

## 3.6 Kết quả CV hiện tại

| Mô hình | F1 CV Spam | Tham số tốt nhất |
|---|---:|---|
| Logistic Regression | 0.9805 | `C=10`, `min_df=2`, `ngram_range=(1,1)` |
| Linear SVM | 0.9821 | `C=1`, `min_df=1`, `ngram_range=(1,2)` |

SVM có điểm CV trung bình cao hơn nhẹ trong lần tìm kiếm này. Khác biệt nhỏ; không nên kết luận SVM luôn tốt hơn trên mọi dữ liệu.

## 3.7 Đầu ra

Sau khi cả hai search chạy thành công, notebook lưu:

- `artifacts/lr_tuned_pipeline.joblib`
- `artifacts/svm_tuned_pipeline.joblib`
- `results/cv_tuning_results.csv`

Pipeline đã gồm TF-IDF và classifier, nên Part 3/dự đoán chỉ cần tải pipeline và đưa văn bản đã clean vào.

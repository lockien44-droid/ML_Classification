# 4. Đánh giá và diễn giải kết quả

Notebook thực hành: [`Part 3 - Evaluation and Interpretation.ipynb`](../notebooks/Part%203%20-%20Evaluation%20and%20Interpretation.ipynb)

## 4.1 Quy tắc đánh giá

Part 3 tải hai pipeline tuned từ `artifacts/`, đọc `test.csv` và tạo dự đoán trên cùng tập email. Test không được dùng lại để thay đổi siêu tham số.

## 4.2 Chỉ số

- **Accuracy:** tỷ lệ dự đoán đúng trên toàn bộ email.
- **Precision Spam:** trong email mô hình gắn nhãn spam, tỷ lệ thực sự là spam. Precision thấp thường nghĩa là nhiều email thường bị chặn nhầm.
- **Recall Spam:** trong email spam thật, tỷ lệ được phát hiện. Recall thấp nghĩa là có nhiều spam bị bỏ sót.
- **F1 Spam:** trung bình điều hòa giữa Precision và Recall; tăng khi cả hai cùng tốt.

Accuracy một mình có thể gây hiểu nhầm khi phân bố lớp lệch. Vì thế dự án báo cáo metrics của lớp Spam riêng.

## 4.3 Kết quả Test hiện tại

Tập Test có 979 email, gồm 687 không spam và 292 spam.

| Mô hình | Accuracy | Precision Spam | Recall Spam | F1 Spam |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.9918 | 0.9897 | 0.9829 | 0.9863 |
| Linear SVM | 0.9959 | 0.9932 | 0.9932 | 0.9932 |

SVM có các chỉ số cao hơn trong phép chia này. Đây là một holdout duy nhất, không phải đảm bảo hiệu năng ngoài thực tế.

## 4.4 Confusion matrix

Hàng là nhãn thật; cột là nhãn dự đoán, thứ tự `[không spam, spam]`.

```text
Logistic Regression:
[[684   3]
 [  5 287]]

Linear SVM:
[[685   2]
 [  2 290]]
```

Diễn giải ma trận LR:

- 684 email thường được nhận đúng là thường.
- 3 email thường bị đánh dấu nhầm là spam (false positive).
- 5 email spam bị bỏ sót, dự đoán thành thường (false negative).
- 287 email spam được phát hiện đúng.

Diễn giải SVM tương tự: 2 false positive và 2 false negative.

## 4.5 So sánh lỗi

Notebook phân nhóm email theo hai dự đoán:

- Cả hai đúng: 971.
- Chỉ LR sai, SVM đúng: 4.
- Chỉ SVM sai, LR đúng: 0.
- Cả hai cùng sai: 4.
- Hai mô hình bất đồng: 4.

Nhóm bất đồng đều thuộc trường hợp LR sai, SVM đúng trong lần chạy này. Có thể đọc ví dụ trong output Part 3 để xem nội dung nào gây nhầm lẫn; không nên suy rộng từ vài ví dụ đơn lẻ.

## 4.6 Đầu ra

- `results/metrics.csv`
- `results/figures/confusion_matrix_lr.png`
- `results/figures/confusion_matrix_svm.png`
- `results/figures/model_metrics_comparison.png`

## 4.7 Xác suất và decision score

Logistic Regression cung cấp `predict_proba()`. Đây là xác suất theo mô hình; nếu cần độ tin cậy xác suất chuẩn, cần hiệu chỉnh và đánh giá calibration.

Linear SVM dùng `decision_function()`. Điểm này biểu thị phía và mức cách biệt tương đối so với ranh giới quyết định. **Không được trình bày điểm đó như xác suất.**

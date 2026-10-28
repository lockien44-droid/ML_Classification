# Linear Support Vector Machine trong dự án

## SVM là gì?

Support Vector Machine tìm ranh giới phân chia lớp sao cho biên giữa hai nhóm được tối đa hóa. “Linear” nghĩa là ranh giới là tuyến tính trong không gian đặc trưng TF-IDF.

Dự án dùng `LinearSVC` của scikit-learn, là classifier; không phải `SVR` hồi quy.

## Vì sao hợp với văn bản?

Email sau TF-IDF thường tạo ma trận thưa với nhiều chiều. Linear SVM thường hoạt động tốt trên kiểu dữ liệu này và không cần chuyển văn bản thành vector đặc dày.

## Siêu tham số `C`

`C` điều chỉnh trade-off giữa biên rộng/regularization và việc phạt lỗi phân loại trên Train. Part 2 thử `C ∈ {0.1, 1.0, 10.0}` qua Stratified 5-fold CV, cùng các cấu hình n-gram/min_df.

Tham số gốc trong `Model/svm.py` là `C=1.0`, `max_iter=20000`, `random_state=42`; sau tuning, pipeline SVM tốt nhất hiện có `C=1.0`, `min_df=1`, `ngram_range=(1,2)`.

## Đầu ra SVM

`LinearSVC.predict()` trả nhãn 0/1. `decision_function()` trả signed decision score liên quan vị trí so với ranh giới. Score dương thường nghiêng về lớp 1 theo thứ tự lớp hiện tại; độ lớn là khoảng cách tương đối, **không phải xác suất**.

## Kết quả hiện tại

- Best CV F1 Spam: `0.9821`.
- Test Accuracy: `0.9959`.
- Test Precision/Recall/F1 Spam: `0.9932 / 0.9932 / 0.9932`.
- Confusion matrix theo `[không spam, spam]`: `[[685, 2], [2, 290]]`.

Đây là một lần chia Test, không đảm bảo hiệu năng tương tự trên mọi bộ email. Xem [Part 3](../notebooks/Part%203%20-%20Evaluation%20and%20Interpretation.ipynb).

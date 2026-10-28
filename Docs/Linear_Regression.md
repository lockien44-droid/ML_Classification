# Linear Regression và Logistic Regression trong bài tập này

## Kết luận trước

Bài toán email spam sử dụng **Logistic Regression**, không sử dụng **Linear Regression**. Hai tên gần giống nhưng mục tiêu khác nhau.

## Linear Regression

- Là thuật toán hồi quy.
- Dự đoán giá trị liên tục, ví dụ giá nhà hoặc điểm năng lượng.
- Đầu ra có thể là bất kỳ số thực nào.

## Logistic Regression

- Dù có chữ “Regression” trong tên, thường được dùng làm thuật toán phân loại.
- Tính xác suất lớp thông qua hàm logistic/sigmoid; với hai lớp, dự đoán một lớp trong `{0,1}`.
- Trong dự án, có thể lấy xác suất spam qua `predict_proba()`.
- Được khởi tạo trong `Model/logistic_regression.py`, rồi dùng trong Pipeline ở Part 2.

## Áp dụng trong dự án

`text` được biến đổi thành TF-IDF; Logistic Regression học trọng số cho các token để phân loại spam/không spam. Không dùng Linear Regression để dự đoán nhãn nhị phân vì đầu ra hồi quy có thể nằm ngoài `[0,1]` và không tối ưu trực tiếp cho phân loại.

Xem thêm [Part 2 — Model Training and Tuning](../notebooks/Part%202%20-%20Model%20Training%20and%20Tuning.ipynb) và [kết quả đánh giá](04_Evaluation_and_Interpretation.md).

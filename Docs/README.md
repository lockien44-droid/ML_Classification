# Tài liệu dự án

Bộ tài liệu này mô tả dự án phân loại email spam từ đầu đến cuối: mục tiêu, dữ liệu, tiền xử lý, chọn mô hình, đánh giá, dự đoán và kiến thức cần trình bày.

## Đọc theo thứ tự

1. [Tổng quan và quy trình A–Z](01_Project_Overview.md)
2. [Khám phá và chuẩn bị dữ liệu](02_Data_Preparation.md)
3. [Huấn luyện và tìm siêu tham số](03_Model_Training_and_Tuning.md)
4. [Đánh giá và phân tích lỗi](04_Evaluation_and_Interpretation.md)
5. [Dự đoán email mới](05_Prediction.md)
6. [Khái niệm, câu hỏi ôn tập và FAQ](06_Concepts_and_FAQ.md)
7. [Logistic Regression và phân biệt với Linear Regression](Linear_Regression.md)
8. [Linear SVM](Support_Vector_Machine.md)

## Notebook và mã nguồn

- Notebook thực hành nằm trong `notebooks/`, chạy Part 1 → Part 2 → Part 3.
- Hàm tái sử dụng nằm trong `src/`.
- Cách cài đặt và chạy ngắn gọn nằm trong [README ở thư mục gốc](../README.md).

Các số liệu ghi trong tài liệu là kết quả đã lưu trong `results/` tại lần chạy gần nhất. Nếu chạy lại notebook trên dữ liệu hoặc môi trường khác, hãy cập nhật số liệu theo output mới.

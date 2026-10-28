# 5. Dự đoán email mới

## 5.1 Điều kiện trước khi dự đoán

Chạy Part 2 ít nhất một lần thành công để có:

- `artifacts/lr_tuned_pipeline.joblib`
- `artifacts/svm_tuned_pipeline.joblib`

## 5.2 Chạy CLI

Từ thư mục gốc dự án, kích hoạt môi trường ảo rồi chạy:

```powershell
python main.py
```

Chọn `1. Dự đoán email mới` và nhập email tiếng Anh. Hoặc bỏ qua menu:

```powershell
python predict.py
```

## 5.3 Luồng bên trong `predict.py`

1. `load_models()` tải hai tuned pipeline.
2. `clean_text()` từ `src/preprocess.py` chuyển chữ thường, bỏ HTML và chuẩn hóa khoảng trắng — cùng quy tắc Part 1.
3. Văn bản sạch được đóng thành một pandas Series tên `text`.
4. Pipeline tự chạy TF-IDF đã fit ở Part 2 rồi gọi classifier.
5. Chương trình hiển thị hai nhãn dự đoán, xác suất spam LR và decision score SVM.

Không fit lại TF-IDF trên email mới. Vectorizer trong pipeline dùng vocabulary/IDF đã học trên Train.

## 5.4 Đọc kết quả

- `Spam`: mô hình dự đoán nhãn `1`.
- `Không spam`: mô hình dự đoán nhãn `0`.
- Xác suất LR: giá trị từ 0 đến 1 được hiển thị thành phần trăm.
- Điểm SVM: điểm quyết định, không phải xác suất và không so sánh trực tiếp theo thang phần trăm.

Nếu nội dung rỗng sau làm sạch, chương trình báo lỗi và không phân loại. Nếu thiếu artifacts, hãy chạy Part 2.

## 5.5 Giới hạn sử dụng

Mô hình học từ bộ dữ liệu tiếng Anh cụ thể. Nội dung ngôn ngữ khác, email rất ngắn, từ cố tình viết sai hoặc chiến thuật spam mới có thể giảm độ chính xác. Kết quả ham/spam không kiểm tra tính an toàn của URL hay tệp đính kèm.

# 6. Khái niệm nền tảng, ôn tập và FAQ

## 6.1 Bảng thuật ngữ

| Thuật ngữ | Ý nghĩa trong dự án |
|---|---|
| Dataset | Tập email và nhãn. |
| Sample/row | Một email. |
| Feature/input `X` | Nội dung `text`, sau đó được TF-IDF chuyển thành số. |
| Target/label `y` | `label_num`: spam `1`, không spam `0`. |
| Binary classification | Bài toán dự đoán một trong hai lớp. |
| Train set | Dữ liệu dùng để fit pipeline và chạy CV. |
| Test set | Dữ liệu giữ riêng để báo cáo hiệu năng cuối. |
| Hyperparameter | Cấu hình chọn trước qua CV, ví dụ `C`, `min_df`, `ngram_range`. |
| Cross-validation | Chia Train thành các fold để đánh giá nhiều lần trong lúc chọn cấu hình. |
| Data leakage | Thông tin từ validation/Test đi vào bước fit hoặc chọn mô hình không đúng quy trình. |
| Pipeline | Chuỗi vectorizer và classifier được fit như một đơn vị. |
| False positive | Email thường bị dự đoán nhầm là spam. |
| False negative | Email spam bị bỏ sót và dự đoán là thường. |

## 6.2 Vì sao không dùng Linear Regression?

Linear Regression dự đoán một giá trị liên tục. Nhãn spam là lớp rời rạc 0/1, nên bài này dùng **Logistic Regression** — một classifier — chứ không dùng **Linear Regression**. Xem [Linear_Regression.md](Linear_Regression.md) để phân biệt tên gọi.

## 6.3 Vì sao dùng `stratify`?

Lớp spam ít hơn lớp không spam. `stratify=label_num` giữ tỷ lệ lớp gần như nhau trong Train và Test, tránh một tập có tỷ lệ lớp khác hẳn toàn bộ dữ liệu.

## 6.4 Vì sao CV chạy trên Train?

CV hỗ trợ chọn siêu tham số mà Test vẫn chưa bị dùng. Sau khi chọn xong, pipeline được fit lại trên toàn bộ Train; chỉ sau đó mới đánh giá Test.

## 6.5 5W1H để trình bày

- **What:** phân loại email spam/không spam bằng Logistic Regression và Linear SVM.
- **Why:** tự động nhận diện spam; so sánh classifier tuyến tính trên đặc trưng văn bản TF-IDF.
- **When:** Part 1 chuẩn bị dữ liệu; Part 2 chọn tham số; Part 3 đánh giá; sau đó dùng CLI dự đoán email mới.
- **Where:** dữ liệu ở `data/`, hàm ở `src/`, notebook ở `notebooks/`, pipeline ở `artifacts/`, kết quả ở `results/`.
- **Who/Which:** mô hình nào tốt hơn được xác định bằng F1 CV trên Train và metrics Test riêng; kết quả hiện tại SVM nhỉnh hơn ở lần chia này.
- **How:** clean text → chia dữ liệu → TF-IDF + classifier trong Pipeline → 5-fold CV → fit best → Test evaluation → inference.

## 6.6 Câu hỏi thường gặp

**Tại sao cần cả CV và Test?**  
CV hỗ trợ chọn tham số trong Train; Test là đánh giá cuối trên dữ liệu chưa tham gia lựa chọn.

**`C` có nghĩa gì?**  
`C` điều chỉnh mức regularization trong classifier. Nó là siêu tham số được chọn trong CV, không phải số lớp.

**Tại sao vectorizer nằm trong Pipeline?**  
Để mỗi fold chỉ học vocabulary/IDF từ fold huấn luyện, tránh nhìn thấy fold validation trước.

**Tại sao accuracy cao nhưng vẫn cần Recall/F1?**  
Accuracy không cho biết loại lỗi nào xảy ra. Với spam, bỏ sót spam và chặn nhầm thư thường có hậu quả khác nhau.

**SVM cho biết spam bao nhiêu phần trăm không?**  
Không. `decision_function` không phải xác suất. Muốn xác suất cần thêm bước calibration được đánh giá riêng.

**Kết quả SVM tốt hơn có nghĩa luôn chọn SVM?**  
Không. Kết quả hiện tại là một lần chia dữ liệu. Với ứng dụng thật cần kiểm tra dữ liệu mới, drift và chi phí của từng loại lỗi.

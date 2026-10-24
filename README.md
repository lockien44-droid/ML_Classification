# Phân loại email spam bằng Logistic Regression và Linear SVM

## 1. Mục tiêu

Xây dựng hai mô hình phân loại email thành:
- 0: Không spam.
- 1: Spam.

So sánh hai mô hình trên cùng tập Test và thử dự đoán email mới.

## 2. Dữ liệu

Nguồn: Spam Mails Dataset trên Kaggle.

https://www.kaggle.com/datasets/venky73/spam-mails-dataset

File gốc:
Data/Raw/spam_ham_dataset.csv

Các cột sử dụng:
- text: Nội dung email.
- label_num: Nhãn số, 0 là không spam, 1 là spam.

Thống kê:
- Dữ liệu gốc: 5.171 email.
- Loại 178 dòng trùng nội dung ban đầu.
- Loại thêm 99 dòng trùng sau làm sạch.
- Dữ liệu cuối: 4.894 email.
- Không spam: 3.432 email.
- Spam: 1.462 email.

## 3. Quy trình thực hiện

1. Kiểm tra dữ liệu thiếu, email trùng và nhãn.
2. Giữ hai cột text và label_num.
3. Loại các bản ghi trùng nội dung.
4. Chuyển văn bản thành chữ thường.
5. Loại thẻ HTML và chuẩn hóa khoảng trắng.
6. Kiểm tra và loại trùng sau làm sạch.
7. Chia Train/Test theo tỷ lệ 80/20:
   - Train: 3.915 email.
   - Test: 979 email.
   - random_state=42.
   - Chia có phân tầng theo nhãn.
8. So sánh ba cấu hình A/B/C bằng Stratified 5-fold CV trên Train; TF-IDF và chuẩn hóa đặc trưng số được fit bên trong từng fold.
9. Chọn cấu hình có F1 lớp Spam trung bình cao nhất riêng cho LR và SVM, rồi fit lại pipeline thắng trên toàn bộ Train.
10. Đánh giá trên Test và dự đoán email mới bằng pipeline đã lưu.

Nhãn đã được mã hóa sẵn trong cột label_num.
Không cần mã hóa lại ham/spam.

## 4. Cài đặt

Mở terminal tại thư mục dự án.

Tạo môi trường ảo nếu chưa có:

```powershell
python -m venv .venv
```

Kích hoạt môi trường trong PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Cài thư viện:

```powershell
python -m pip install -r requirements.txt
```

## 5. Chạy chương trình

```powershell
python main.py
```

Các lựa chọn:
- 1: Dự đoán email mới.
- 2: Đánh giá và so sánh hai mô hình.
- 3: Huấn luyện hai mô hình.
- 4: Tiền xử lý và chia dữ liệu.
- 0: Thoát.

Khi chạy lần đầu, chọn theo thứ tự:
4, 3, 2, 1.

Sau khi đã chạy bước 4 và bước 3, có thể chọn trực tiếp 1 hoặc 2.

Chạy riêng từng bước:

```powershell
python preprocess.py
python train.py
python evaluate.py
python predict.py
```

Khi thay đổi dữ liệu hoặc cách làm sạch,
cần huấn luyện lại trước khi đánh giá và dự đoán.

## 6. Chức năng các file

| File | Chức năng |
|---|---|
| preprocess.py | Làm sạch, loại trùng, chia và lưu dữ liệu |
| Model/logistic_regression.py | Khởi tạo Logistic Regression |
| Model/Svm.py | Khởi tạo Linear SVM |
| Model/features.py | Cấu hình A/B/C và trích xuất đặc trưng số |
| train.py | Chạy CV, chọn cấu hình, fit lại và lưu pipeline |
| evaluate.py | Tính chỉ số, vẽ confusion matrix và phân tích lỗi |
| predict.py | Dự đoán email mới |
| main.py | Menu điều phối các chức năng |
| requirements.txt | Danh sách thư viện cần cài |

predict.py sử dụng chung hàm clean_text từ preprocess.py
để giữ quy tắc xử lý nhất quán.

## 7. Các file được tạo

Data/processed/:
- emails_clean.csv
- train.csv
- test.csv

artifacts/:
- lr_pipeline.joblib
- svm_pipeline.joblib

results/:
- cv_results.csv
- metrics.csv
- confusion_matrix_lr.png
- confusion_matrix_svm.png
- prediction_comparison.csv

## 8. Kết quả thực nghiệm

CV dùng Stratified 5-fold, shuffle và random_state=42. Số liệu dưới đây là F1 lớp Spam trung bình ± độ lệch chuẩn; tham số mô hình được khai báo trong `Model/logistic_regression.py` và `Model/svm.py`.

| Mô hình | A: TF-IDF unigram | B: TF-IDF unigram + bigram | C: B + đặc trưng số | Chọn |
|---|---:|---:|---:|---|
| LR | 97,21% ± 0,65% | 96,10% ± 0,98% | 92,16% ± 0,79% | A |
| Linear SVM | 97,80% ± 0,53% | 98,21% ± 0,61% | 98,01% ± 0,57% | B |

C bổ sung số từ, URL, dấu `!`, dấu `$` và tỷ lệ chữ hoa trên nội dung gốc. `preprocess.py` giữ cột `raw_text` để không mất thông tin chữ hoa; các đặc trưng số được chuẩn hóa riêng trong mỗi fold.

Precision, Recall và F1 trong bảng Test bên dưới được tính cho lớp spam. Tham số SVM dùng `C=1.0`, `max_iter=20000`, `random_state=42`.

| Mô hình | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Logistic Regression (A) | 99,28% | 98,63% | 98,97% | 98,80% |
| Linear SVM (B) | 99,59% | 99,32% | 99,32% | 99,32% |

Confusion matrix có hàng là nhãn thật,
cột là nhãn dự đoán, theo thứ tự [Không spam, Spam].

Logistic Regression:

```text
[[683   4]
 [  3 289]]
```

Linear SVM:

```text
[[685   2]
 [  2 290]]
```

Đối chiếu lỗi:
- Hai mô hình cùng sai: 3 email.
- LR sai, SVM đúng: 4 email.
- SVM sai, LR đúng: 1 email.

Trên tập Test này, SVM có ít lỗi hơn LR.

## 9. Lưu ý và hạn chế

- Dữ liệu và email demo sử dụng tiếng Anh.
- Kết quả trên thuộc một lần chia Train/Test.
- Email ngắn, từ viết biến dạng hoặc nội dung quảng cáo
  hợp lệ có thể gây khó khăn cho mô hình.
- Xác suất spam chỉ được hiển thị cho Logistic Regression.
- Điểm quyết định của Linear SVM không phải xác suất.
- Nhãn spam/không spam không xác định độ an toàn
  của liên kết hoặc file đính kèm.

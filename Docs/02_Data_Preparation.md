# 2. Khám phá và chuẩn bị dữ liệu

Notebook thực hành: [`Part 1 - Data Preparation.ipynb`](../notebooks/Part%201%20-%20Data%20Preparation.ipynb)

## 2.1 Nguồn và cấu trúc dữ liệu

Nguồn là [Spam Mails Dataset trên Kaggle](https://www.kaggle.com/datasets/venky73/spam-mails-dataset). File đặt tại `data/raw/spam_ham_dataset.csv`.

CSV gốc có 5.171 hàng và 4 cột:

- `Unnamed: 0`: chỉ số từ file nguồn; không dùng làm đặc trưng.
- `label`: nhãn dạng chữ `ham`/`spam`; chỉ dùng khi kiểm tra mô tả.
- `text`: nội dung email — đầu vào văn bản.
- `label_num`: nhãn số — mục tiêu cần dự đoán (`0` ham, `1` spam).

Notebook giữ `text`, `label_num` và thêm `raw_text`, là bản nội dung nguyên casing trước khi làm sạch. `raw_text` hữu ích cho các bước EDA/đặc trưng cũ; pipeline tuned hiện tại dùng cột `text`.

## 2.2 Kiểm tra chất lượng

Part 1 hiển thị kích thước, tên cột, số thiếu theo cột, phân bố nhãn và số nội dung trùng. Trong lần chạy hiện tại không có giá trị thiếu. Nếu dữ liệu mới có hàng thiếu `text` hoặc `label_num`, hàm `prepare_dataset()` loại chúng trước khi huấn luyện có giám sát.

Phân bố ban đầu:

| Nhãn | Số lượng |
|---|---:|
| Không spam (`0`) | 3.672 |
| Spam (`1`) | 1.499 |

Hai lớp không cân bằng hoàn toàn; do đó Part 1 và cross-validation dùng phân tầng, còn đánh giá báo cáo thêm Precision/Recall/F1.

## 2.3 Làm sạch văn bản

Hàm `clean_text()` trong `src/preprocess.py` thực hiện ba bước:

1. Đổi chữ về lowercase.
2. Thay thẻ HTML dạng `<...>` bằng khoảng trắng.
3. Gộp khoảng trắng liên tiếp và xóa khoảng trắng ở hai đầu.

Ví dụ:

```text
"<b>Limited</b> OFFER   now"
→ "limited offer now"
```

Không xóa dấu câu hoặc số. TF-IDF và tokenization của scikit-learn xử lý phần văn bản sau đó.

## 2.4 Loại trùng

- Loại trùng theo nội dung gốc trước khi clean: 178 email.
- Làm sạch rồi kiểm tra trùng lần nữa: phát hiện và loại 99 email.
- Còn 4.894 email.

Cần kiểm tra sau làm sạch vì hai email khác nhau ở casing/HTML/khoảng trắng có thể trở thành cùng một chuỗi. Notebook cũng kiểm tra nội dung trùng giữa Train và Test; trong lần chạy hiện tại số giao bằng 0.

## 2.5 Chia Train/Test

Hàm `split_dataset()` gọi `train_test_split()` với:

```python
train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label_num"],
)
```

Kết quả hiện tại:

| Tập | Số email | Tỷ lệ |
|---|---:|---:|
| Train | 3.915 | 80% |
| Test | 979 | 20% |

`stratify` giữ tỷ lệ lớp gần như nhau ở hai tập. `random_state=42` giúp có thể lặp lại phép chia trên cùng dữ liệu/thứ tự dòng. Nó không đảm bảo kết quả giữ nguyên nếu dataset thay đổi.

## 2.6 Vì sao giữ Test riêng?

Train được dùng để fit mô hình và chọn siêu tham số bằng CV. Test mô phỏng dữ liệu chưa thấy và chỉ được dùng ở Part 3 cho báo cáo cuối. Nếu xem điểm Test để thử nhiều tham số rồi chọn cấu hình theo điểm đó, Test không còn là đánh giá độc lập.

## 2.7 Đầu ra và cách tạo lại

Hàm `save_prepared_datasets()` lưu:

- `data/processed/emails_clean.csv`
- `data/processed/train.csv`
- `data/processed/test.csv`

Chạy lại Part 1 sẽ ghi lại các file này. Nếu muốn giữ kết quả cũ để so sánh, hãy sao lưu trước khi chạy lại.

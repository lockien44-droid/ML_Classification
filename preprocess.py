from pathlib import Path
import re

import pandas as pd
from sklearn.model_selection import train_test_split


# Đường dẫn dùng trong bước tiền xử lý
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Data" / "Raw" / "spam_ham_dataset.csv"
PROCESSED_DIR = BASE_DIR / "Data" / "processed"


# Hàm làm sạch dùng chung cho dữ liệu và email mới
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    # 1. Đọc dữ liệu gốc
    df = pd.read_csv(DATA_PATH)

    print("Kích thước dữ liệu:", df.shape)
    print("Các cột:", df.columns.tolist())

    print(df[["label", "label_num"]].head())

    print("Email đầu tiên:")
    print(df["text"].iloc[0][:300])

    # 2. Kiểm tra chất lượng dữ liệu
    print("\nSố giá trị thiếu trong mỗi cột:")
    print(df.isna().sum())

    print("\nSố email của từng nhãn:")
    print(df["label"].value_counts())

    print("\nSố email trùng nội dung:")
    print(df.duplicated(subset=["text"]).sum())

    # 3. Giữ các cột cần thiết và loại trùng ban đầu
    # Keep the original casing for the numeric uppercase-ratio feature.
    df = df[["text", "label_num"]].copy()
    df["raw_text"] = df["text"]

    rows_before = len(df)

    df = df.drop_duplicates(subset=["text"], keep="first")
    df = df.reset_index(drop=True)

    print("\nSố dòng đã loại:", rows_before - len(df))
    print("Kích thước sau loại trùng:", df.shape)

    print("Số email mỗi nhãn sau loại trùng:")
    print(df["label_num"].value_counts())

    # 4. Làm sạch nội dung email
    sample_before = df["text"].iloc[0]

    df["text"] = df["text"].apply(clean_text)

    print("\nTrước làm sạch:")
    print(sample_before[:300])

    print("\nSau làm sạch:")
    print(df["text"].iloc[0][:300])

    # 5. Kiểm tra sau làm sạch
    print("\nSố email rỗng sau làm sạch:")
    print(df["text"].eq("").sum())

    print("\nSố email trùng sau làm sạch:")
    print(df.duplicated(subset=["text"]).sum())

    print("\nSố nội dung có nhãn mâu thuẫn:")
    label_counts = df.groupby("text")["label_num"].nunique()
    print(label_counts.gt(1).sum())

    # 6. Loại trùng phát sinh sau làm sạch
    df = df.drop_duplicates(subset=["text"], keep="first")
    df = df.reset_index(drop=True)

    print("\nKích thước dữ liệu sau làm sạch và loại trùng:")
    print(df.shape)

    print("\nSố email còn lại của mỗi nhãn:")
    print(df["label_num"].value_counts())

    print("\nSố dòng còn trùng:")
    print(df.duplicated(subset=["text"]).sum())

    # 7. Lưu dữ liệu đã làm sạch
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    clean_path = PROCESSED_DIR / "emails_clean.csv"
    df.to_csv(clean_path, index=False, encoding="utf-8")

    print("\nĐã lưu dữ liệu tại:", clean_path)

    # 8. Chia Train/Test theo tỷ lệ 80/20
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label_num"]
    )

    # 9. Lưu hai tập dữ liệu
    train_df.to_csv(
        PROCESSED_DIR / "train.csv",
        index=False,
        encoding="utf-8"
    )

    test_df.to_csv(
        PROCESSED_DIR / "test.csv",
        index=False,
        encoding="utf-8"
    )
        # Lấy những nội dung xuất hiện trong cả Train và Test
    shared_texts = set(train_df["text"]) & set(test_df["text"])

    print("\nSố nội dung trùng giữa Train và Test:", len(shared_texts))

    if shared_texts:
        raise ValueError("Train và Test có nội dung trùng nhau.")

    # 10. Kiểm tra kết quả chia
    print("\nKích thước Train:", train_df.shape)
    print("Kích thước Test:", test_df.shape)

    print("\nTỷ lệ nhãn trong Train:")
    print(train_df["label_num"].value_counts(normalize=True))

    print("\nTỷ lệ nhãn trong Test:")
    print(test_df["label_num"].value_counts(normalize=True))


# Chỉ tự chạy khi thực thi trực tiếp preprocess.py
if __name__ == "__main__":
    main()

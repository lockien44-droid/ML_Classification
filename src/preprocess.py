from pathlib import Path
import re

import pandas as pd
from sklearn.model_selection import train_test_split


# Đường dẫn dùng trong bước tiền xử lý
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "spam_ham_dataset.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"


# Hàm làm sạch dùng chung cho dữ liệu và email mới
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def prepare_dataset(df):
    """Select, clean, and deduplicate email data; return it with diagnostics."""
    required_columns = {"text", "label_num"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            "Thiếu cột bắt buộc: " + ", ".join(sorted(missing_columns))
        )

    prepared = df[["text", "label_num"]].copy()
    before_missing = len(prepared)
    prepared = prepared.dropna(subset=["text", "label_num"]).copy()
    missing_rows_removed = before_missing - len(prepared)

    prepared["text"] = prepared["text"].astype(str)
    # Keep original casing for numeric email features used by model pipelines.
    prepared["raw_text"] = prepared["text"]

    before_raw_dedup = len(prepared)
    prepared = prepared.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
    raw_duplicates_removed = before_raw_dedup - len(prepared)

    sample_before = prepared["text"].iloc[0] if not prepared.empty else ""
    prepared["text"] = prepared["text"].map(clean_text)
    empty_after_cleaning = int(prepared["text"].eq("").sum())
    cleaned_duplicates = int(prepared.duplicated(subset=["text"]).sum())

    label_counts = prepared.groupby("text")["label_num"].nunique()
    conflicting_labels = int(label_counts.gt(1).sum())

    before_cleaned_dedup = len(prepared)
    prepared = prepared.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
    cleaned_duplicates_removed = before_cleaned_dedup - len(prepared)

    diagnostics = {
        "missing_rows_removed": missing_rows_removed,
        "raw_duplicates_removed": raw_duplicates_removed,
        "cleaned_duplicates": cleaned_duplicates,
        "cleaned_duplicates_removed": cleaned_duplicates_removed,
        "empty_after_cleaning": empty_after_cleaning,
        "conflicting_labels": conflicting_labels,
        "sample_before": sample_before,
        "sample_after": prepared["text"].iloc[0] if not prepared.empty else "",
    }
    return prepared, diagnostics


def split_dataset(df, test_size=0.2, random_state=42):
    """Create stratified train/test data while keeping feature-label rows aligned."""
    return train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["label_num"],
    )


def save_prepared_datasets(df, train_df, test_df, output_dir=PROCESSED_DIR):
    """Save cleaned, train, and test CSV files and return their paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "clean": output_dir / "emails_clean.csv",
        "train": output_dir / "train.csv",
        "test": output_dir / "test.csv",
    }
    df.to_csv(paths["clean"], index=False, encoding="utf-8")
    train_df.to_csv(paths["train"], index=False, encoding="utf-8")
    test_df.to_csv(paths["test"], index=False, encoding="utf-8")
    return paths

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

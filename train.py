from pathlib import Path

import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer

from Model.logistic_regression import create_model as create_lr_model
from Model.svm import create_model as create_svm_model


# Đường dẫn dữ liệu và nơi lưu mô hình
BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "Data" / "processed"
ARTIFACTS_DIR = BASE_DIR / "artifacts"


def main():
    # 1. Đọc dữ liệu huấn luyện
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")

    X_train = train_df["text"]
    y_train = train_df["label_num"]

    print("Số email Train:", len(X_train))
    print("Số nhãn Train:", len(y_train))

    # 2. Học TF-IDF và biến đổi tập Train
    vectorizer = TfidfVectorizer(stop_words="english")
    # Học từ vựng và IDF chỉ trên Train
    X_train_tfidf = vectorizer.fit_transform(X_train)
    example = "the meeting is about the project"

    print("\nCác từ giữ lại trong câu ví dụ:")
    print(vectorizer.build_analyzer()(example))

    print("Kích thước TF-IDF Train:", X_train_tfidf.shape)

    print("Kích thước TF-IDF Train:", X_train_tfidf.shape)

    # 3. Huấn luyện Logistic Regression
    lr_model = create_lr_model()
    lr_model.fit(X_train_tfidf, y_train)

    print("Đã huấn luyện Logistic Regression.")

    # 4. Huấn luyện Linear SVM
    svm_model = create_svm_model()
    svm_model.fit(X_train_tfidf, y_train)

    print("Đã huấn luyện Linear SVM.")

    # 5. Tạo thư mục lưu
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # 6. Lưu TF-IDF và hai mô hình
    joblib.dump(
        vectorizer,
        ARTIFACTS_DIR / "tfidf_vectorizer.joblib"
    )

    joblib.dump(
        lr_model,
        ARTIFACTS_DIR / "logistic_regression.joblib"
    )

    joblib.dump(
        svm_model,
        ARTIFACTS_DIR / "svm.joblib"
    )

    print("\nĐã lưu TF-IDF và hai mô hình tại:", ARTIFACTS_DIR)


if __name__ == "__main__":
    main()
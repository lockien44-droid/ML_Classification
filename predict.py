from pathlib import Path
from preprocess import clean_text
import joblib


# Đường dẫn đến thư mục chứa các thành phần đã học
BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"




# Chức năng dự đoán email
def main():
    # 1. Tải TF-IDF và hai mô hình
    vectorizer = joblib.load(
        ARTIFACTS_DIR / "tfidf_vectorizer.joblib"
    )

    lr_model = joblib.load(
        ARTIFACTS_DIR / "logistic_regression.joblib"
    )

    svm_model = joblib.load(
        ARTIFACTS_DIR / "svm.joblib"
    )

    print("Đã tải TF-IDF và hai mô hình.")
    print(
        "Số đặc trưng TF-IDF:",
        len(vectorizer.get_feature_names_out())
    )
    print("Các lớp Logistic Regression:", lr_model.classes_)
    print("Các lớp SVM:", svm_model.classes_)

    # 2. Nhập và làm sạch email
    email = input("\nNhập nội dung email bằng tiếng Anh: ")
    cleaned_email = clean_text(email)

    if not cleaned_email:
        print("Email không được để trống.")
        return

    # 3. Biến đổi email bằng TF-IDF đã học
    email_tfidf = vectorizer.transform([cleaned_email])
        # Kiểm tra email có đặc trưng được TF-IDF nhận diện không
    if email_tfidf.nnz == 0:
        print(
            "Không tìm thấy từ nào thuộc bộ từ vựng đã học. "
            "Vui lòng nhập email tiếng Anh có nội dung rõ hơn."
        )
        return

    # 4. Dự đoán nhãn
    lr_prediction = lr_model.predict(email_tfidf)[0]
    svm_prediction = svm_model.predict(email_tfidf)[0]

    # 5. Tính xác suất spam của Logistic Regression
    spam_column = list(lr_model.classes_).index(1)

    lr_probability = lr_model.predict_proba(
        email_tfidf
    )[0, spam_column]

    # 6. Tính điểm quyết định của SVM
    svm_score = svm_model.decision_function(email_tfidf)[0]

    # 7. Chuyển nhãn số thành chữ
    labels = {
        0: "Không spam",
        1: "Spam"
    }

    # 8. Hiển thị kết quả
    print("\nEmail sau làm sạch:", cleaned_email)

    print("Logistic Regression:", labels[lr_prediction])
    print(f"Xác suất spam của LR: {lr_probability:.2%}")

    print("Linear SVM:", labels[svm_prediction])
    print(f"Điểm quyết định SVM: {svm_score:.4f}")


# Chỉ tự chạy khi thực thi trực tiếp file predict.py
if __name__ == "__main__":
    main()
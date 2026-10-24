from pathlib import Path

import joblib
import pandas as pd

from preprocess import clean_text


# Đường dẫn đến thư mục chứa các thành phần đã học
BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"




# Chức năng dự đoán email
def main():
    # 1. Tải pipeline gồm tiền xử lý đặc trưng và bộ phân loại
    lr_model = joblib.load(ARTIFACTS_DIR / "lr_pipeline.joblib")
    svm_model = joblib.load(ARTIFACTS_DIR / "svm_pipeline.joblib")
    print("Đã tải hai pipeline LR và SVM.")

    # 2. Nhập và làm sạch email
    email = input("\nNhập nội dung email bằng tiếng Anh: ")
    cleaned_email = clean_text(email)

    if not cleaned_email:
        print("Email không được để trống.")
        return

    email_features = pd.DataFrame([{
        "text": cleaned_email,
        "raw_text": email,
    }])

    # 4. Dự đoán nhãn
    lr_prediction = lr_model.predict(email_features)[0]
    svm_prediction = svm_model.predict(email_features)[0]

    # 5. Tính xác suất spam của Logistic Regression
    spam_column = list(lr_model.classes_).index(1)

    lr_probability = lr_model.predict_proba(email_features)[0, spam_column]

    # 6. Tính điểm quyết định của SVM
    svm_score = svm_model.decision_function(email_features)[0]

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

from pathlib import Path

import joblib
import pandas as pd

from src.preprocess import clean_text


BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
LABELS = {0: "Không spam", 1: "Spam"}


def load_models():
    """Load the tuned text pipelines produced by Part 2."""
    model_paths = {
        "Logistic Regression": ARTIFACTS_DIR / "lr_tuned_pipeline.joblib",
        "Linear SVM": ARTIFACTS_DIR / "svm_tuned_pipeline.joblib",
    }
    missing = [str(path) for path in model_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Không tìm thấy pipeline đã tune: " + ", ".join(missing)
            + ". Hãy chạy notebook Part 2 trước."
        )
    return {name: joblib.load(path) for name, path in model_paths.items()}


def predict_email(email, models=None):
    """Clean one email and return predictions from both saved pipelines."""
    cleaned_email = clean_text(email)
    if not cleaned_email:
        raise ValueError("Email không được để trống.")

    if models is None:
        models = load_models()

    # Part 2 trained on the cleaned text column as a one-dimensional input.
    text_input = pd.Series([cleaned_email], name="text")
    lr_model = models["Logistic Regression"]
    svm_model = models["Linear SVM"]

    lr_prediction = int(lr_model.predict(text_input)[0])
    svm_prediction = int(svm_model.predict(text_input)[0])

    spam_column = list(lr_model.classes_).index(1)
    lr_spam_probability = float(
        lr_model.predict_proba(text_input)[0, spam_column]
    )
    svm_decision_score = float(svm_model.decision_function(text_input)[0])

    return {
        "cleaned_email": cleaned_email,
        "lr_prediction": lr_prediction,
        "lr_spam_probability": lr_spam_probability,
        "svm_prediction": svm_prediction,
        "svm_decision_score": svm_decision_score,
    }


def main():
    models = load_models()
    print("Đã tải pipeline đã tune cho Logistic Regression và Linear SVM.")
    email = input("\nNhập nội dung email bằng tiếng Anh: ")

    try:
        result = predict_email(email, models=models)
    except ValueError as error:
        print(error)
        return

    print("\nEmail sau làm sạch:", result["cleaned_email"])
    print("Logistic Regression:", LABELS[result["lr_prediction"]])
    print(f"Xác suất spam của LR: {result['lr_spam_probability']:.2%}")
    print("Linear SVM:", LABELS[result["svm_prediction"]])
    print(
        "Điểm quyết định SVM: "
        f"{result['svm_decision_score']:.4f} (không phải xác suất)"
    )


if __name__ == "__main__":
    main()

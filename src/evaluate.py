from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# Đường dẫn các thư mục
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
RESULTS_DIR = BASE_DIR / "results"


def main():
    # 1. Đọc tập Test đã được làm sạch
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")
    if "raw_text" not in test_df.columns:
        raise ValueError("Thiếu raw_text trong test.csv. Hãy chạy preprocess.py trước.")

    X_test = test_df[["text", "raw_text"]]
    y_test = test_df["label_num"]

    # 2. Tải các pipeline đã chọn cấu hình bằng CV
    lr_model = joblib.load(ARTIFACTS_DIR / "lr_pipeline.joblib")
    svm_model = joblib.load(ARTIFACTS_DIR / "svm_pipeline.joblib")

    print("Số email Test:", len(X_test))
    print("Số nhãn Test:", len(y_test))
    print("Cấu hình LR:", _pipeline_configuration(lr_model))
    print("Cấu hình SVM:", _pipeline_configuration(svm_model))
    print("Đã tải hai pipeline đã fit.")

    # 4. Chuẩn bị nơi lưu kết quả và danh sách mô hình
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    models = {
        "Logistic Regression": lr_model,
        "Linear SVM": svm_model
    }

    image_names = {
        "Logistic Regression": "confusion_matrix_lr.png",
        "Linear SVM": "confusion_matrix_svm.png"
    }

    results = []
    predictions_by_model = {}

    # 5. Dự đoán và đánh giá từng mô hình
    for name, model in models.items():
        predictions = model.predict(X_test)

        # Lưu dự đoán để phân tích lại mà không cần predict lần nữa
        predictions_by_model[name] = predictions

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(
                y_test,
                predictions,
                pos_label=1,
                zero_division=0
            ),
            "Recall": recall_score(
                y_test,
                predictions,
                pos_label=1,
                zero_division=0
            ),
            "F1": f1_score(
                y_test,
                predictions,
                pos_label=1,
                zero_division=0
            )
        })

        cm = confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1]
        )

        print(f"\nConfusion matrix của {name}:")
        print(cm)

        # 6. Vẽ và lưu confusion matrix
        fig, ax = plt.subplots(figsize=(6, 5))

        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Not spam", "Spam"]
        )

        display.plot(
            ax=ax,
            cmap="Blues",
            values_format="d",
            colorbar=False
        )

        ax.set_title(name)
        fig.tight_layout()

        image_path = RESULTS_DIR / image_names[name]
        fig.savefig(image_path, dpi=200)
        plt.close(fig)

        print("Đã lưu ảnh:", image_path)

    # 7. Hiển thị và lưu bảng chỉ số
    results_df = pd.DataFrame(results)

    print("\nBảng so sánh hai mô hình:")
    print(
        results_df.to_string(
            index=False,
            float_format="%.4f"
        )
    )

    metrics_path = RESULTS_DIR / "metrics.csv"

    results_df.to_csv(
        metrics_path,
        index=False,
        encoding="utf-8"
    )

    print("\nĐã lưu bảng đánh giá tại:", metrics_path)

    # 8. Xem ba email Logistic Regression dự đoán sai
    lr_predictions = predictions_by_model["Logistic Regression"]

    error_analysis = test_df[["text", "label_num"]].copy()
    error_analysis["prediction"] = lr_predictions

    wrong_emails = error_analysis[
        error_analysis["label_num"] != error_analysis["prediction"]
    ]

    print("\nSố email LR dự đoán sai:", len(wrong_emails))
    print("\nBa email LR dự đoán sai đầu tiên:")

    labels = {
        0: "Không spam",
        1: "Spam"
    }

    for index, row in wrong_emails.head(3).iterrows():
        print(f"\nVị trí trong bảng Test: {index}")
        print("Nhãn thật:", labels[row["label_num"]])
        print("Nhãn dự đoán:", labels[row["prediction"]])
        print("Nội dung:", row["text"][:500])

    # 9. MỚI: tạo bảng đối chiếu dự đoán của hai mô hình
    comparison = test_df[["text", "label_num"]].copy()

    comparison["lr_prediction"] = predictions_by_model[
        "Logistic Regression"
    ]

    comparison["svm_prediction"] = predictions_by_model[
        "Linear SVM"
    ]

    # Đánh dấu các dòng mỗi mô hình dự đoán sai
    lr_wrong = (
        comparison["lr_prediction"] != comparison["label_num"]
    )

    svm_wrong = (
        comparison["svm_prediction"] != comparison["label_num"]
    )

    # 10. MỚI: phân loại các trường hợp lỗi
    both_wrong = comparison[lr_wrong & svm_wrong]
    only_lr_wrong = comparison[lr_wrong & ~svm_wrong]
    only_svm_wrong = comparison[~lr_wrong & svm_wrong]

    print("\nĐối chiếu lỗi giữa hai mô hình:")
    print("Cả hai cùng sai:", len(both_wrong))
    print("LR sai, SVM đúng:", len(only_lr_wrong))
    print("SVM sai, LR đúng:", len(only_svm_wrong))

    # 11. MỚI: lưu dự đoán của toàn bộ email Test để tra cứu
    comparison_path = RESULTS_DIR / "prediction_comparison.csv"

    comparison.to_csv(
        comparison_path,
        index=True,
        index_label="test_row",
        encoding="utf-8"
    )

    print("Đã lưu bảng đối chiếu tại:", comparison_path)
        # Gom hai nhóm cần xem để dùng chung cách hiển thị
    error_groups = {
        "LR sai, SVM đúng": only_lr_wrong,
        "SVM sai, LR đúng": only_svm_wrong
    }

    for group_name, group_df in error_groups.items():
        print(f"\n===== {group_name} =====")

        for index, row in group_df.iterrows():
            print(f"\nVị trí trong bảng Test: {index}")
            print("Nhãn thật:", labels[row["label_num"]])
            print("LR dự đoán:", labels[row["lr_prediction"]])
            print("SVM dự đoán:", labels[row["svm_prediction"]])
            print("Nội dung:", row["text"][:500])


def _pipeline_configuration(pipeline):
    features = pipeline.named_steps["features"]
    transformers = features.named_transformers_
    if "numeric" in transformers:
        return "C"
    if transformers["tfidf"].ngram_range == (1, 2):
        return "B"
    return "A"


if __name__ == "__main__":
    main()

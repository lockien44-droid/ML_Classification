from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from Model.features import FEATURE_CONFIGS, create_feature_transformer
from Model.logistic_regression import create_model as create_lr_model
from Model.svm import create_model as create_svm_model


BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
RESULTS_DIR = BASE_DIR / "results"
CV_RANDOM_STATE = 42


def create_pipeline(config_name, model):
    return Pipeline([
        ("features", create_feature_transformer(config_name)),
        ("model", clone(model)),
    ])


def main():
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    required_columns = {"text", "raw_text", "label_num"}
    missing_columns = required_columns - set(train_df.columns)
    if missing_columns:
        raise ValueError(
            "Thiếu cột trong train.csv: " + ", ".join(sorted(missing_columns))
            + ". Hãy chạy preprocess.py để tạo lại dữ liệu có raw_text."
        )

    X_train = train_df[["text", "raw_text"]]
    y_train = train_df["label_num"]
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=CV_RANDOM_STATE,
    )
    scoring = {
        "f1_spam": "f1",
        "precision_spam": "precision",
        "recall_spam": "recall",
    }
    models = {
        "LR": create_lr_model(),
        "SVM": create_svm_model(),
    }

    print(f"Train: {len(X_train)} email; Stratified 5-fold CV; seed={CV_RANDOM_STATE}")
    print("Đang so sánh A/B/C bằng F1 lớp Spam...\n")

    rows = []
    best_by_model = {}
    for model_name, model in models.items():
        model_candidates = []
        for config_name, config in FEATURE_CONFIGS.items():
            pipeline = create_pipeline(config_name, model)
            scores = cross_validate(
                pipeline,
                X_train,
                y_train,
                cv=cv,
                scoring=scoring,
                return_train_score=False,
                n_jobs=1,
            )
            f1_scores = scores["test_f1_spam"]
            row = {
                "Model": model_name,
                "Configuration": config_name,
                "Description": config["description"],
                "F1_mean": f1_scores.mean(),
                "F1_std": f1_scores.std(ddof=1),
                "Precision_mean": scores["test_precision_spam"].mean(),
                "Recall_mean": scores["test_recall_spam"].mean(),
                "F1_folds": ";".join(f"{score:.6f}" for score in f1_scores),
            }
            rows.append(row)
            model_candidates.append(row)
            print(
                f"{model_name} {config_name} ({config['description']}): "
                f"F1={row['F1_mean']:.4f} ± {row['F1_std']:.4f}; "
                f"Precision={row['Precision_mean']:.4f}; "
                f"Recall={row['Recall_mean']:.4f}"
            )

        winner = max(model_candidates, key=lambda item: item["F1_mean"])
        best_by_model[model_name] = winner

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cv_results_path = RESULTS_DIR / "cv_results.csv"
    pd.DataFrame(rows).to_csv(cv_results_path, index=False, encoding="utf-8")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    for model_name, model in models.items():
        winner = best_by_model[model_name]
        pipeline = create_pipeline(winner["Configuration"], model)
        pipeline.fit(X_train, y_train)
        artifact_name = "lr_pipeline.joblib" if model_name == "LR" else "svm_pipeline.joblib"
        artifact_path = ARTIFACTS_DIR / artifact_name
        joblib.dump(pipeline, artifact_path)
        print(
            f"\n{model_name} thắng cấu hình {winner['Configuration']} "
            f"(F1={winner['F1_mean']:.4f} ± {winner['F1_std']:.4f}); "
            f"đã fit lại trên toàn bộ Train và lưu {artifact_path}"
        )

    print(f"\nBảng kết quả CV: {cv_results_path}")


if __name__ == "__main__":
    main()

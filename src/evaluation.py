from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def calculate_metrics(y_true, predictions):
    """Return per-model classification metrics and confusion matrices."""
    metric_rows = []
    confusion_matrices = {}
    for model_name, y_pred in predictions.items():
        metric_rows.append({
            "Model": model_name,
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision (Spam)": precision_score(
                y_true, y_pred, pos_label=1, zero_division=0
            ),
            "Recall (Spam)": recall_score(
                y_true, y_pred, pos_label=1, zero_division=0
            ),
            "F1 (Spam)": f1_score(
                y_true, y_pred, pos_label=1, zero_division=0
            ),
        })
        confusion_matrices[model_name] = confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        )
    return pd.DataFrame(metric_rows), confusion_matrices


def save_confusion_matrix_plots(confusion_matrices, output_dir, label_names):
    """Save one labeled confusion-matrix figure per model and return figures/paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = {}
    for model_name, matrix in confusion_matrices.items():
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        image = ax.imshow(matrix, cmap="Blues")
        ax.set_title(f"Confusion Matrix — {model_name}")
        ax.set_xlabel("Nhãn dự đoán")
        ax.set_ylabel("Nhãn thật")
        ax.set_xticks([0, 1], [label_names[0], label_names[1]])
        ax.set_yticks([0, 1], [label_names[0], label_names[1]])
        threshold = matrix.max() / 2 if matrix.max() else 0
        for row in range(2):
            for column in range(2):
                ax.text(
                    column,
                    row,
                    f"{matrix[row, column]:,}",
                    ha="center",
                    va="center",
                    color="white" if matrix[row, column] > threshold else "black",
                    fontsize=12,
                )
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        filename = (
            "confusion_matrix_lr.png"
            if model_name == "Logistic Regression"
            else "confusion_matrix_svm.png"
        )
        output_path = output_dir / filename
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        figures[model_name] = (fig, output_path)
    return figures


def save_metric_comparison(metrics_df, output_dir):
    """Save a bar chart comparing accuracy and spam-focused metrics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_columns = [
        "Accuracy",
        "Precision (Spam)",
        "Recall (Spam)",
        "F1 (Spam)",
    ]
    ax = metrics_df.set_index("Model")[metric_columns].plot(
        kind="bar", figsize=(9, 5), ylim=(0, 1), rot=0
    )
    ax.set_title("So sánh Logistic Regression và Linear SVM trên Test")
    ax.set_ylabel("Điểm số")
    ax.set_xlabel("")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.25)
    fig = ax.get_figure()
    fig.tight_layout()
    output_path = output_dir / "model_metrics_comparison.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    return fig, output_path

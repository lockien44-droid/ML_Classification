import matplotlib
matplotlib.use("QtAgg")

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.widgets import Button
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# 1. TẠO DATASET

X, y = make_moons(
    n_samples=400,
    noise=0.25,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# 2. HUẤN LUYỆN HAI MÔ HÌNH

lr_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

svm_model = make_pipeline(
    StandardScaler(),
    SVC(kernel="rbf", C=1.0, gamma="scale")
)

lr_model.fit(X_train, y_train)
svm_model.fit(X_train, y_train)

# 3. DỰ ĐOÁN VÀ ĐÁNH GIÁ

lr_pred = lr_model.predict(X_test)
svm_pred = svm_model.predict(X_test)

lr_accuracy = accuracy_score(y_test, lr_pred)
svm_accuracy = accuracy_score(y_test, svm_pred)

print(f"Logistic Regression Accuracy: {lr_accuracy:.2%}")
print(f"SVM Accuracy: {svm_accuracy:.2%}")


# 4. HÀM VẼ DECISION BOUNDARY

def draw_boundary(ax, model, title):

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 250),
        np.linspace(y_min, y_max, 250)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    predictions = model.predict(grid).reshape(xx.shape)

    ax.contourf(
        xx, yy, predictions,
        levels=[-0.5, 0.5, 1.5],
        colors=["#b9d9f3", "#f8c5bd"],
        alpha=0.55
    )

    ax.contour(
        xx, yy, predictions,
        levels=[0.5],
        colors="black",
        linewidths=1.5
    )

    ax.scatter(
        X_test[:, 0],
        X_test[:, 1],
        c=y_test,
        cmap="coolwarm",
        edgecolors="black",
        s=35
    )

    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")


# 5. GIAO DIỆN 3 MÀN HÌNH

fig = plt.figure(figsize=(12, 7))
current_page = 0


def show_page(page):

    global current_page
    current_page = page

    fig.clear()

    # Chừa khoảng trống phía dưới cho nút
    fig.subplots_adjust(
        left=0.08,
        right=0.95,
        top=0.86,
        bottom=0.19,
        wspace=0.3
    )

    if page == 0:

        ax = fig.add_subplot(111)

        ax.scatter(
            X_train[:, 0],
            X_train[:, 1],
            c=y_train,
            cmap="coolwarm",
            edgecolors="black",
            s=45
        )

        ax.set_title(
            "1. Dataset and Logistic Regression Training",
            fontsize=16
        )

        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")

        ax.text(
            0.02, 0.02,
            "Dataset: make_moons\n"
            "Total: 400 samples\n"
            "Training: 300 samples\n"
            "Testing: 100 samples\n"
            "Model: Logistic Regression",
            transform=ax.transAxes,
            fontsize=11,
            bbox=dict(facecolor="white", alpha=0.9)
        )

    elif page == 1:

        ax = fig.add_subplot(111)

        draw_boundary(
            ax,
            lr_model,
            f"2. Logistic Regression | Accuracy: {lr_accuracy:.2%}"
        )

        sample = X_test[0].reshape(1, -1)

        probability = lr_model.predict_proba(sample)[0, 1]
        predicted_class = lr_model.predict(sample)[0]

        ax.scatter(
            sample[0, 0],
            sample[0, 1],
            s=220,
            facecolors="none",
            edgecolors="green",
            linewidths=3
        )

        ax.text(
            0.02, 0.02,
            f"Sample prediction:\n"
            f"P(y=1) = {probability:.2%}\n"
            f"Predicted class = {predicted_class}\n"
            f"Threshold = 0.5\n"
            f"Sigmoid(z) = 1 / (1 + exp(-z))",
            transform=ax.transAxes,
            fontsize=11,
            bbox=dict(facecolor="white", alpha=0.9)
        )

    elif page == 2:

        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        draw_boundary(
            ax1,
            lr_model,
            f"Logistic Regression\nAccuracy: {lr_accuracy:.2%}"
        )

        draw_boundary(
            ax2,
            svm_model,
            f"SVM (RBF Kernel)\nAccuracy: {svm_accuracy:.2%}"
        )

        sample = X_test[0].reshape(1, -1)
        svm_class = svm_model.predict(sample)[0]

        ax2.scatter(
            sample[0, 0],
            sample[0, 1],
            s=220,
            facecolors="none",
            edgecolors="green",
            linewidths=3
        )

        fig.suptitle(
            f"3. SVM Prediction: Class {svm_class} | "
            "Logistic Regression vs SVM",
            fontsize=16
        )

    # Tạo nút điều hướng
    ax_prev = fig.add_axes([0.32, 0.05, 0.14, 0.065])
    ax_next = fig.add_axes([0.54, 0.05, 0.14, 0.065])

    btn_prev = Button(ax_prev, "Previous")
    btn_next = Button(ax_next, "Next")

    def previous(event):
        if current_page > 0:
            show_page(current_page - 1)

    def next_page(event):
        if current_page < 2:
            show_page(current_page + 1)

    btn_prev.on_clicked(previous)
    btn_next.on_clicked(next_page)

    # Giữ tham chiếu để nút không bị thu gom bộ nhớ
    fig._buttons = (btn_prev, btn_next)

    fig.canvas.draw_idle()


# 6. CHẠY DEMO

show_page(0)
plt.show()
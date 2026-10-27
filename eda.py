from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Model.features import count_url_markers, extract_numeric_features


BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "Data" / "processed"
EDA_DIR = BASE_DIR / "results" / "eda"
LABEL_NAMES = {0: "Ham", 1: "Spam"}
LABEL_COLORS = {0: "#4C78A8", 1: "#F58518"}


def _load_train_data():
    train_path = PROCESSED_DIR / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {train_path}. Hãy chạy preprocess.py trước."
        )

    train_df = pd.read_csv(train_path)
    required_columns = {"text", "raw_text", "label_num"}
    missing_columns = required_columns - set(train_df.columns)
    if missing_columns:
        raise ValueError(
            "Thiếu cột trong train.csv: " + ", ".join(sorted(missing_columns))
            + ". Hãy chạy preprocess.py để tạo lại dữ liệu có raw_text."
        )
    return train_df


def _build_summary(train_df):
    raw_texts = train_df["raw_text"].fillna("").astype(str).to_numpy()
    numeric_features = extract_numeric_features(raw_texts)
    numeric_df = pd.DataFrame(
        numeric_features,
        columns=[
            "word_count_model",
            "url_marker_count",
            "exclamation_count",
            "dollar_count",
            "uppercase_ratio",
        ],
        index=train_df.index,
    )

    # Keep the slide's word-count definition: whitespace-separated words.
    word_counts = train_df["text"].fillna("").astype(str).str.split().str.len()
    url_markers = train_df["raw_text"].fillna("").astype(str).map(count_url_markers)
    exclamation_counts = train_df["raw_text"].fillna("").astype(str).str.count("!")
    dollar_counts = train_df["raw_text"].fillna("").astype(str).map(lambda text: text.count("$"))

    summary_rows = []
    for label in (0, 1):
        mask = train_df["label_num"].eq(label)
        label_words = word_counts[mask]
        label_urls = url_markers[mask]
        label_exclamations = exclamation_counts[mask]
        label_dollars = dollar_counts[mask]
        label_uppercase = numeric_df.loc[mask, "uppercase_ratio"]
        email_count = int(mask.sum())

        summary_rows.append({
            "Label": LABEL_NAMES[label],
            "Email_count": email_count,
            "Email_share_pct": 100 * email_count / len(train_df),
            "Words_Q1": label_words.quantile(0.25),
            "Words_median": label_words.median(),
            "Words_Q3": label_words.quantile(0.75),
            "Emails_with_url_markers": int(label_urls.gt(0).sum()),
            "URL_marker_email_rate_pct": 100 * label_urls.gt(0).mean(),
            "Mean_url_markers": label_urls.mean(),
            "Emails_with_exclamation": int(label_exclamations.gt(0).sum()),
            "Exclamation_email_rate_pct": 100 * label_exclamations.gt(0).mean(),
            "Mean_exclamation_count": label_exclamations.mean(),
            "Median_exclamation_count": label_exclamations.median(),
            "Emails_with_dollar": int(label_dollars.gt(0).sum()),
            "Dollar_email_rate_pct": 100 * label_dollars.gt(0).mean(),
            "Mean_dollar_count": label_dollars.mean(),
            "Uppercase_ratio_mean_pct": 100 * label_uppercase.mean(),
            "Uppercase_ratio_median_pct": 100 * label_uppercase.median(),
        })

    return pd.DataFrame(summary_rows), word_counts, url_markers, exclamation_counts, dollar_counts


def _save_label_distribution(train_df):
    counts = train_df["label_num"].value_counts().reindex([0, 1], fill_value=0)
    total = len(train_df)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(
        [LABEL_NAMES[label] for label in counts.index],
        counts.values,
        color=[LABEL_COLORS[label] for label in counts.index],
        width=0.58,
    )
    ax.set_title("Nhãn email trong tập Train")
    ax.set_ylabel("Số email")
    ax.set_ylim(0, max(counts.values) * 1.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, count in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts.values) * 0.025,
            f"{count:,}\n{count / total:.2%}",
            ha="center",
            va="bottom",
        )
    fig.tight_layout()
    fig.savefig(EDA_DIR / "label_distribution.png", dpi=200)
    plt.close(fig)


def _save_word_length_plot(train_df, word_counts):
    values = [word_counts[train_df["label_num"].eq(label)] for label in (0, 1)]
    fig, ax = plt.subplots(figsize=(7, 4.8))
    box = ax.boxplot(
        values,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#222222", "linewidth": 2},
    )
    ax.set_xticks([1, 2], [LABEL_NAMES[label] for label in (0, 1)])
    for patch, label in zip(box["boxes"], (0, 1)):
        patch.set_facecolor(LABEL_COLORS[label])
        patch.set_alpha(0.75)
    ax.set_title("Độ dài email trong tập Train")
    ax.set_ylabel("Số từ (tách theo khoảng trắng)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(EDA_DIR / "email_length_distribution.png", dpi=200)
    plt.close(fig)


def _save_signal_rates_plot(train_df, url_markers, exclamation_counts, dollar_counts):
    signal_rates = {}
    for label in (0, 1):
        mask = train_df["label_num"].eq(label)
        signal_rates[LABEL_NAMES[label]] = [
            100 * url_markers[mask].gt(0).mean(),
            100 * exclamation_counts[mask].gt(0).mean(),
            100 * dollar_counts[mask].gt(0).mean(),
        ]

    x = np.arange(3)
    width = 0.34
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for offset, label in [(-width / 2, "Ham"), (width / 2, "Spam")]:
        bars = ax.bar(
            x + offset,
            signal_rates[label],
            width,
            label=label,
            color=LABEL_COLORS[0 if label == "Ham" else 1],
        )
        ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
    ax.set_xticks(x, ["Có dấu hiệu URL", "Có dấu !", "Có dấu $"])
    ax.set_ylabel("Tỷ lệ email")
    ax.set_ylim(0, max(max(values) for values in signal_rates.values()) * 1.25 + 1)
    ax.set_title("Tỷ lệ dấu hiệu trong email theo nhãn")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(EDA_DIR / "email_signal_rates.png", dpi=200)
    plt.close(fig)


def main():
    train_df = _load_train_data()
    EDA_DIR.mkdir(parents=True, exist_ok=True)

    summary, word_counts, url_markers, exclamation_counts, dollar_counts = _build_summary(train_df)
    _save_label_distribution(train_df)
    _save_word_length_plot(train_df, word_counts)
    _save_signal_rates_plot(train_df, url_markers, exclamation_counts, dollar_counts)

    summary_path = EDA_DIR / "eda_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8")
    print("EDA summary for Train:")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    print(f"\nSaved charts and summary to: {EDA_DIR}")


if __name__ == "__main__":
    main()

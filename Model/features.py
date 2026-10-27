import re

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


FEATURE_CONFIGS = {
    "A": {
        "description": "TF-IDF từ đơn",
        "ngram_range": (1, 1),
        "numeric": False,
    },
    "B": {
        "description": "TF-IDF từ đơn và từ đôi",
        "ngram_range": (1, 2),
        "numeric": False,
    },
    "C": {
        "description": "TF-IDF từ đơn và từ đôi + đặc trưng số",
        "ngram_range": (1, 2),
        "numeric": True,
    },
}


# Detect link-related markers even when the source dataset has spaces or other
# formatting that prevents a complete URL from matching. This counts markers,
# not validated URLs, so it is named accordingly throughout the project.
URL_MARKER_PATTERN = re.compile(r"\b(?:https?|www)\b", flags=re.IGNORECASE)


def count_url_markers(text):
    """Count http/https/www markers in original email text."""
    if text is None:
        return 0
    return len(URL_MARKER_PATTERN.findall(str(text)))


def extract_numeric_features(raw_texts):
    """Extract email style counts from the original, case-preserving text."""
    texts = np.asarray(raw_texts, dtype=object).reshape(-1)
    rows = []

    for value in texts:
        text = "" if value is None else str(value)
        letters = [char for char in text if char.isalpha()]
        uppercase_ratio = (
            sum(char.isupper() for char in letters) / len(letters)
            if letters else 0.0
        )
        rows.append([
            len(re.findall(r"\b\w+\b", text)),
            count_url_markers(text),
            text.count("!"),
            text.count("$"),
            uppercase_ratio,
        ])

    return np.asarray(rows, dtype=float)


def create_feature_transformer(config_name):
    if config_name not in FEATURE_CONFIGS:
        raise ValueError(f"Cấu hình không hợp lệ: {config_name}")

    config = FEATURE_CONFIGS[config_name]
    transformers = [
        (
            "tfidf",
            TfidfVectorizer(
                stop_words="english",
                ngram_range=config["ngram_range"],
            ),
            "text",
        )
    ]

    if config["numeric"]:
        numeric_pipeline = Pipeline([
            (
                "extract",
                FunctionTransformer(
                    extract_numeric_features,
                    validate=False,
                ),
            ),
            ("scale", StandardScaler()),
        ])
        transformers.append(("numeric", numeric_pipeline, "raw_text"))

    return ColumnTransformer(transformers)

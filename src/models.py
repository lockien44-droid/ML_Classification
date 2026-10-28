from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from Model.logistic_regression import create_model as create_lr_model
from Model.svm import create_model as create_svm_model


# Parameters are searched using F1 for the positive (spam=1) class.
TUNING_PARAM_GRID = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__min_df": [1, 2],
    "classifier__C": [0.1, 1.0, 10.0],
}


def create_text_pipeline(classifier):
    """Build a text-classification pipeline with TF-IDF inside each CV fold."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("classifier", classifier),
    ])


def create_model_pipelines():
    """Return fresh Logistic Regression and Linear SVM text pipelines."""
    return {
        "Logistic Regression": create_text_pipeline(create_lr_model()),
        "Linear SVM": create_text_pipeline(create_svm_model()),
    }

from sklearn.linear_model import LogisticRegression


def create_model():
    return LogisticRegression(max_iter=1000)
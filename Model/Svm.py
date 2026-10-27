from sklearn.svm import LinearSVC


def create_model():
    return LinearSVC(C=1.0, max_iter=5000, random_state=42)



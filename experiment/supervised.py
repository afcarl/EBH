import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import (
    QuadraticDiscriminantAnalysis as QDA,
    LinearDiscriminantAnalysis as LDA
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from EBH.utility.operation import load_dataset


class ClassifierMock:

    def __init__(self):
        self.categ = []

    def fit(self, X, Y):
        self.categ = np.unique(Y)

    def predict(self, X):
        return np.random.choice(self.categ, size=X.shape[0])


def _split(alpha, X, Y, shuffle=True):
    N = len(X)
    tN = int(N*alpha)
    arg = np.arange(N)
    if shuffle:
        np.random.shuffle(arg)
    targ, larg = arg[:tN], arg[tN:]
    return X[larg], Y[larg], X[targ], Y[targ]


def _test_model(model, modelname, X, Y, repeats=1, split=0.1, verbose=1):
    acc = np.empty(repeats)
    for r in range(1, repeats+1):
        lX, lY, tX, tY = _split(split, X, Y)
        model.fit(lX, lY)
        a = (model.predict(tX) == tY).mean()
        acc[r-1] = a
    if verbose:
        print(f"{modelname} accuracy: {acc.mean():.2%}")
    return acc.mean()


def run_classical_models():
    X, Y = load_dataset(as_matrix=True, normalize=True)
    w = "balanced"
    for model, name in [
        (ClassifierMock(), "Baseline (Random)"),
        (LogisticRegression(class_weight=w), "Logistic Regression"),
        (LDA(), "LDA"), (QDA(), "QDA"),
        (GaussianNB(), "Naive Bayes"),
        (KNeighborsClassifier(), "K-Nearest Neighbours"),
        (RandomForestClassifier(class_weight=w), "Random Forest"),
        (SVC(kernel="linear", class_weight=w), "Linear SVM"),
        (SVC(kernel="rbf", class_weight=w), "RBF-SVM")
    ]:
        _test_model(model, name, X, Y, repeats=100, verbose=2)


if __name__ == '__main__':
    run_classical_models()
    # fit_ann()

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import (
    QuadraticDiscriminantAnalysis as QDA
)

from EBH.utility.operation import load_dataset, split_data
from EBH.utility.const import ltbroot


class ClassifierMock:

    def __init__(self):
        self.categ = []

    def fit(self, X, Y):
        self.categ = np.unique(Y)

    def predict(self, X):
        return np.random.choice(self.categ, size=X.shape[0])


def _test_model_crossval(model, modelname, X, Y, repeats=1, split=0.1, verbose=1):
    acc = np.empty(repeats)
    for r in range(1, repeats+1):
        lX, lY, vX, vY = split_data(X, Y, split)
        model.fit(lX, lY)
        a = (model.predict(vX) == vY).mean()
        acc[r-1] = a
    if verbose:
        print(f"{modelname} accuracy: {acc.mean():.2%}")
    return acc.mean()


def _test_model_leaveout(model, modelname, verbose=1):
    lX, lY = load_dataset(ltbroot + "learning.pkl.gz", as_matrix=True, normalize=True)
    model.fit(lX, lY)
    tX, tY = load_dataset(ltbroot + "testing.pkl.gz", as_matrix=True, normalize=True)
    print(f"{modelname} accuracy: {(model.predict(tX) == tY).mean():.2%}")


def run_classical_models(test):
    w = "balanced"
    svmC = 1.0
    for model, name in [
        # (ClassifierMock(), "Baseline (Random)"),
        # (LogisticRegression(class_weight=w), "Logistic Regression"),
        # (QDA(), "Quadratic Discriminant Analysis"),
        # (GaussianNB(), "Naive Bayes"),
        (KNeighborsClassifier(weights="distance"), "K-Nearest Neighbours"),
        # (RandomForestClassifier(class_weight=w), "Random Forest"),
        (SVC(kernel="linear", class_weight=w, C=svmC), "Linear SVM"),
        (SVC(kernel="rbf", class_weight=w, C=svmC), "RBF-SVM"),
        (SVC(kernel="poly", degree=4, C=svmC), "Poly (4) SVM")
    ]:
        test(model, name)


if __name__ == '__main__':
    run_classical_models(test=_test_model_leaveout)

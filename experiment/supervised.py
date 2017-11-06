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

from EBH.utility.operation import load_dataset, as_onehot


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
    strln = len(str(repeats))
    acc = np.empty(repeats)
    print("-" * 50)
    for r in range(1, repeats+1):
        lX, lY, tX, tY = _split(split, X, Y)
        model.fit(lX, lY)
        a = (model.predict(tX) == tY).mean()
        acc[r-1] = a
        if verbose > 1:
            print(f"\rTesting {modelname} round {r:>{strln}}/{repeats}, Acc: {a:.4f}", end="")
    if verbose:
        print(f"\n{modelname} final accuracy: {acc.mean():.4f}")
    return acc.mean()


def run_classical_models():
    X, Y = load_dataset(as_matrix=True, normalize=True)
    for model, name in [
        (ClassifierMock(), "Baseline (Random)"),
        (LogisticRegression(), "Logistic Regression"),
        (LDA(), "LDA"), (QDA(), "QDA"),
        (GaussianNB(), "Naive Bayes"),
        (KNeighborsClassifier(), "K-Nearest Neighbours"),
        (RandomForestClassifier(), "Random Forest"),
        (SVC(kernel="linear"), "Linear SVM"),
        (SVC(kernel="rbf"), "RBF-SVM")
    ]:
        _test_model(model, name, X, Y, repeats=100, verbose=2)


def fit_ann():
    # from sklearn.utils import compute_class_weight
    from tensorflow.contrib import keras as K

    from EBH.utility.visual import plot_learning_dynamics

    Sequential = K.models.Sequential
    Dense, BN = K.layers.Dense, K.layers.BatchNormalization

    lX, ly, tX, ty = load_dataset(0.1, as_matrix=True, normalize=True)
    lY, tY = as_onehot(ly, ty)

    ann = Sequential([
        Dense(input_dim=lX.shape[1], units=120, activation="relu"), BN(),
        Dense(units=60, activation="relu"), BN(),
        Dense(units=30, activation="tanh"), BN(),
        Dense(units=lY.shape[1], activation="softmax")
    ])
    ann.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["acc"])

    history = ann.fit(lX, lY, epochs=100, validation_data=(tX, tY), shuffle=True, verbose=0)
    plot_learning_dynamics(history)


if __name__ == '__main__':
    run_classical_models()

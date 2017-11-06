from EBH.utility.operation import load_dataset, as_onehot


def do_lda():
    from matplotlib import pyplot as plt
    from sklearn.discriminant_analysis import (
        LinearDiscriminantAnalysis as LDA
    )
    from csxdata.visual.scatter import Scatter2D
    X, Y = load_dataset(as_matrix=True, as_string=True)
    model = LDA().fit(X, Y)
    evar = model.explained_variance_ratio_

    print(f"EXPLAINED VARIANCE: {evar}, TOTAL: {evar.sum()}")
    scat = Scatter2D(model.transform(X)[:, (1, 2)], Y, title="LDA transform", axlabels=("DF01", "DF02"))
    scat.split_scatter()
    plt.legend()
    plt.show()


def random_classifier_mock():
    from random import choice
    lX, lY, tX, tY = load_dataset(0.1)
    categ = list(set(tY))

    eq = [choice(categ) == label for label in tY]

    print("Random baseline accuracy:", sum(eq) / len(eq))


def fit_svm(kernel="linear"):
    from sklearn.svm import SVC
    lX, lY, tX, tY = load_dataset(split=0.1, as_matrix=True, normalize=True)

    svm = SVC(kernel=kernel)
    svm.fit(lX, lY)
    acc = (svm.predict(tX) == tY).sum() / len(tX)
    print(f"Support Vector Machine with {kernel} kernel:", acc)


def fit_ann():
    # from sklearn.utils import compute_class_weight
    from tensorflow.contrib import keras as K

    from EBH.utility.visual import plot_learning_dynamics

    Sequential = K.models.Sequential
    Dense, BN = K.layers.Dense, K.layers.BatchNormalization

    X, y = load_dataset(as_matrix=True, normalize=True)
    Y = as_onehot(y)
    # w = compute_class_weight("balanced", np.unique(y), y)

    ann = Sequential([
        Dense(input_dim=X.shape[1], units=120, activation="relu"), BN(),
        Dense(units=60, activation="relu"), BN(),
        Dense(units=30, activation="tanh"), BN(),
        Dense(units=Y.shape[1], activation="softmax")
    ])
    ann.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["acc"])

    history = ann.fit(X, Y, epochs=100, validation_split=0.1, shuffle=True, verbose=0)
    plot_learning_dynamics(history)


def fit_logregress():
    from sklearn.linear_model import LogisticRegression

    lX, lY, tX, tY = load_dataset(0.1, as_matrix=True, normalize=True)

    model = LogisticRegression()
    model.fit(lX, lY)

    print("Logistic regression acc:", (model.predict(tX) == tY).mean())


def fit_naive_bayes():
    from sklearn.naive_bayes import GaussianNB
    lX, lY, tX, tY = load_dataset(split=0.1, as_matrix=True, normalize=True)

    model = GaussianNB()
    model.fit(lX, lY)

    print("Gaussian NB acc:", (model.predict(tX) == tY).mean())


def fit_knn():
    from sklearn.neighbors import KNeighborsClassifier

    lX, lY, tX, tY = load_dataset(0.2, as_matrix=True, normalize=True)

    knn = KNeighborsClassifier()
    knn.fit(lX, lY)

    print("KNN acc:", (knn.predict(tX) == tY).mean())


def fit_random_forest():
    from sklearn.ensemble import RandomForestClassifier

    lX, lY, tX, tY = load_dataset(0.1, as_matrix=True, normalize=True)

    forest = RandomForestClassifier()
    forest.fit(lX, lY)

    print("Random forest acc:", (forest.predict(tX) == tY).mean())


if __name__ == '__main__':
    # do_lda()
    random_classifier_mock()
    fit_svm("linear")
    fit_svm("rbf")
    fit_svm("sigmoid")
    fit_ann()
    fit_naive_bayes()
    fit_knn()
    fit_random_forest()
    fit_logregress()

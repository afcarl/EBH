from EBH.utility.operation import load_dataset


def do_lda(X, Y):
    from matplotlib import pyplot as plt
    from sklearn.discriminant_analysis import (
        LinearDiscriminantAnalysis as LDA
    )
    from csxdata.visual.scatter import Scatter2D

    model = LDA().fit(X, Y)
    evar = model.explained_variance_ratio_

    print(f"EXPLAINED VARIANCE: {evar}, TOTAL: {evar.sum()}")
    scat = Scatter2D(model.transform(X)[:, :2], Y, title="LDA transform")
    scat.split_scatter()
    plt.legend()
    plt.show()


def fit_svm(kernel="linear"):
    from sklearn.svm import SVC
    lX, lY, tX, tY = load_dataset(split=0.1)
    svm = SVC(kernel=kernel)
    svm.fit(lX, lY)
    acc = (svm.predict(tX) == tY).sum() / len(tX)
    print(f"Support Vector Machine with {kernel} kernel:", acc)


if __name__ == '__main__':
    fit_svm("linear")
    fit_svm("rbf")
    fit_svm("sigmoid")

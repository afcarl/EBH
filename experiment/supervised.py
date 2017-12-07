import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from csxdata.utilities.vectorop import split_by_categories

from EBH.utility.operation import load_testsplit_dataset, as_matrix
from EBH.utility.const import boxer_names


class ClassifierMock:

    def __init__(self):
        self.categ = []

    def fit(self, X, Y):
        self.categ = np.unique(Y)

    def predict(self, X):
        return np.random.choice(self.categ, size=X.shape[0])


def load(boxer):

    def resplit(x):
        ax, ay, az = x.T
        return as_matrix(np.concatenate((ax.T, ay.T, np.abs(ay.T), az.T), axis=-1))

    lX, lY, tX, tY = load_testsplit_dataset(boxer, as_matrix=False, as_string=True)
    lX, tX = map(resplit, (lX, tX))
    return lX, lY, tX, tY


# noinspection PyTypeChecker
def split_eval(model, tX, tY):
    splitarg = split_by_categories(tY)
    accs = dict()
    for cat in splitarg:
        arg = splitarg[cat]
        accs[cat] = (np.mean(model.predict(tX[arg]) == tY[arg]), len(arg))
    accs["ALL"] = (np.mean(model.predict(tX) == tY), len(tY))
    return accs


def xperiment_leave_one_out(modeltype, initarg: dict=None, verbosity=1):
    accs = []
    nonj = []
    initarg = dict() if initarg is None else initarg  # type: dict
    for name in boxer_names:
        lX, lY, tX, tY = load(name)
        model = modeltype(**initarg)
        if verbosity > 1:
            print("-"*50)
            print(f"{model.__class__.__name__} vs E_{name}")
        model.fit(lX, lY)
        bycat_acc = split_eval(model, tX, tY)
        if verbosity > 1:
            for cat in ("J", "U", "H", "ALL"):
                acc, n = bycat_acc[cat]
                print(f"{cat} ({n}): {acc:.2%}")
        accs.append(bycat_acc["ALL"][0])
        nonj.append((bycat_acc["H"][0] + bycat_acc["U"][0])/2.)
    if verbosity:
        print("*"*50)
        print(f"MODEL: {modeltype.__name__.upper()} - {initarg}")
        print(f"OVERALL ACCURACY: {np.mean(accs):.2%}")
        print(f"OVERALL NON-J ACCURACY: {np.mean(nonj):.2%}")


LOADERARG = dict(as_matrix=True, as_string=True, optimalish=True, drop_outliers=0.95)


if __name__ == '__main__':
    # xperiment_leave_one_out(ClassifierMock)
    # xperiment_leave_one_out(SVC, dict(C=.1337, kernel="poly", degree=2, class_weight="balanced"))  # 73%
    # xperiment_leave_one_out(SVC, dict(C=.1337, kernel="rbf", class_weight="balanced"))  # 66%
    # xperiment_leave_one_out(RandomForestClassifier, dict(class_weight="balanced"))  # 67%
    xperiment_leave_one_out(KNeighborsClassifier, dict(n_neighbors=5, metric="manhattan"))  # 65%
    # xperiment_leave_one_out(GaussianNB)  # 63%
    # xperiment_leave_one_out(QDA)  # 68%
    # xperiment_leave_one_out(MLPClassifier, dict(learning_rate_init=0.1))  # 65%
    # xperiment_leave_one_out(Perceptron, dict(class_weight="balanced", max_iter=1000, tol=1e-3))

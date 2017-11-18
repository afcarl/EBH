import numpy as np
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.neural_network import MLPClassifier

from csxdata.utilities.vectorop import split_by_categories

from EBH.utility.operation import load_testsplit_dataset
from EBH.utility.const import boxer_names


class ClassifierMock:

    def __init__(self):
        self.categ = []

    def fit(self, X, Y):
        self.categ = np.unique(Y)

    def predict(self, X):
        return np.random.choice(self.categ, size=X.shape[0])


# noinspection PyTypeChecker
def split_eval(model, tX, tY):
    splitarg = split_by_categories(tY)
    accs = dict()
    for cat in "JUH":
        arg = splitarg[cat]
        accs[cat] = np.mean(model.predict(tX[arg]) == tY[arg])
    accs["ALL"] = np.mean(model.predict(tX) == tY)
    return accs


def xperiment_leave_one_out(model):
    accs = []
    for name in boxer_names:
        lX, lY, tX, tY = load_testsplit_dataset(name, as_matrix=True, normalize=True, as_string=True)
        print("-"*50)
        print(f"{model.__class__.__name__} vs E_{name}")
        model.fit(lX, lY)
        bycat_acc = split_eval(model, tX, tY)
        for cat in ("J", "U", "H", "ALL"):
            print(f"{cat}: {bycat_acc[cat]:.2%}")
        accs.append(bycat_acc["ALL"])
    print("*"*50)
    print("TOTAL OVERALL ACCURACIES:", np.mean(accs))


if __name__ == '__main__':
    # xperiment_leave_one_out(SVC(C=0.1, kernel="poly", degree=2, class_weight="balanced"))  # 73%
    # xperiment_leave_one_out(RandomForestClassifier(n_jobs=4, class_weight="balanced"))  # 67%
    # xperiment_leave_one_out(KNeighborsClassifier(n_jobs=5))  # 71%
    # xperiment_leave_one_out(GaussianNB())  # 60%
    # xperiment_leave_one_out(QDA())
    xperiment_leave_one_out(MLPClassifier(verbose=False, warm_start=True, learning_rate_init=0.1))

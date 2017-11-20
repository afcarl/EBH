import numpy as np

from sklearn.neighbors import KNeighborsClassifier

from EBH.utility.operation import load_testsplit_dataset, shuffle


def salt_data(X1, Y1, X2, Y2, ratio):
    salt = int(len(X2) * ratio)
    X2, Y2 = shuffle(X2, Y2)
    return np.r_[X1, X2[:salt]], np.r_[Y1, Y2[:salt]], X2[salt:], Y2[salt:]


data = load_testsplit_dataset("Virginia", as_matrix=True, as_string=True)
for r in np.arange(0.01, 1., 0.01):
    lX, lY, tX, tY = salt_data(*data, ratio=r)
    knn = KNeighborsClassifier().fit(lX, lY)
    print(f"KNN vs {len(data[2])-len(tY)} salted data: {(knn.predict(tX) == tY).mean():.2%}")

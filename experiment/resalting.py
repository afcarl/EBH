import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier as KNN

from EBH.utility.operation import load_testsplit_dataset, shuffle


def salt_data(X1, Y1, X2, Y2, ratio):
    salt = int(len(X2) * ratio)
    X2, Y2 = shuffle(X2, Y2)
    return np.r_[X1, X2[:salt]], np.r_[Y1, Y2[:salt]], X2[salt:], Y2[salt:]


data = load_testsplit_dataset("Virginia", as_matrix=True, as_string=True)
curveX, curveY = [], []
for r in np.arange(0.00, 1., 0.01):
    lX, lY, tX, tY = salt_data(*data, ratio=r)
    knn = KNN().fit(lX, lY)
    curveX.append(len(data[2])-len(tY))
    curveY.append((knn.predict(tX) == tY).mean())
    print(f"KNN vs {curveX[-1]:>4} salted data: {curveY[-1]:.2%}")

plt.plot(curveX, curveY)
plt.grid()
plt.show()

import numpy as np
from matplotlib import pyplot as plt

from EBH.utility.const import clproot
from EBH.utility.frame import DataWrapper


def clapsumplot(dw: DataWrapper):
    left, right = dw.get_data("left", norm=False), dw.get_data("right", norm=False)
    left, right = np.abs(left).sum(axis=1) / 3, np.abs(right).sum(axis=1) / 3
    combo = np.maximum(left, right)
    fig, (tx, mx, bx) = plt.subplots(3, 1, sharex=True, sharey=True)
    tx.plot(left)
    tx.grid()
    mx.plot(right)
    mx.grid()
    bx.plot(combo)
    bx.grid()
    plt.show()


clapsumplot(DataWrapper(clproot + "Istvan_fel.txt"))

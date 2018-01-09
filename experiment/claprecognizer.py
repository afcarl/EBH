import numpy as np
from matplotlib import pyplot as plt

from EBH.utility.const import clproot
from EBH.utility.frame import DataWrapper


def clapsumplot(dw: DataWrapper):
    left, right = dw.get_data("left", norm=False), dw.get_data("right", norm=False)
    left, right = np.abs(left).sum(axis=1) / 3, np.abs(right).sum(axis=1) / 3
    combo = np.maximum(left, right)
    fig, axarr = plt.subplots(3, 1, sharex=True, sharey=True)
    for title, data, ax in zip(["left", "right", "combo"], [left, right, combo], axarr):
        ax.plot(data)
        ax.grid()
        ax.set_title(title)
    ax.plot([40 for _ in range(len(data))], "r--", alpha=.5)
    plt.show()


clapsumplot(DataWrapper(clproot + "Istvan_fel.txt"))

import numpy as np
from matplotlib import pyplot as plt

from EBH.utility.const import clproot
from EBH.utility.frame import DataWrapper
from EBH.utility.operation import average_filter


def clapsumplot(dw: DataWrapper):
    left, right = dw.get_data("left", norm=True), dw.get_data("right", norm=True)
    sumdata = (left + right) / 2
    fig, (tx, bx) = plt.subplots(2, 1, sharex=True, sharey=True)
    tx.plot(dw.time, left)
    tx.plot(dw.time, right)
    tx.grid()
    bx.plot(dw.time, sumdata)
    bx.grid()
    plt.show()


clapsumplot(DataWrapper(clproot + "Istvan1_clap.txt"))

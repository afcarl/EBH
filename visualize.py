import numpy as np
from matplotlib import pyplot as plt

from utility.peak import find_peaks
from utility.frame import DataWrapper

from EBH import logroot


def plot_acceleration(dw, show=True, dumppath=None):
    ltime, ldata, rtime, rdata = dw.data
    fig, axarr = plt.subplots(2, 2, figsize=(20, 10))
    for i, (lcol, rcol, ax) in enumerate(zip(ldata.T, rdata.T, axarr.flat[:3]), start=1):
        ax.set_title(f"Axis {i}")
        ax.plot(ltime, lcol)
        ax.plot(rtime, rcol)
        ax.grid()
    axarr[-1, -1].plot(ltime, np.linalg.norm(ldata, axis=1), label="Left")
    axarr[-1, -1].plot(rtime, np.linalg.norm(rdata, axis=1), label="Right")
    axarr[-1, -1].set_title("Vector size (L2 norm)")
    axarr[-1, -1].grid()
    plt.suptitle("Time vs. signal intensity")
    plt.figlegend()
    plt.tight_layout()
    if dumppath:
        plt.savefig(dumppath)
    if show:
        plt.show()
    else:
        plt.clf()
        plt.close()


def plot_peaks(time, data, thresh, ax=None, title=None):
    ax = plt.gca() if ax is None else ax
    Y = np.linalg.norm(data, axis=1)
    peaks = find_peaks(Y)
    ax.plot(time, Y)
    ax.plot(time[peaks], Y[peaks], "rx")
    ax.plot(time, np.ones_like(time)*thresh, "r--")
    ax.set_title(title)
    ax.grid()
    return ax


def main():
    import os
    for file in sorted(os.listdir(logroot)):
        dw = DataWrapper(logroot + file)
        fig, (tax, bax) = plt.subplots(2, 1, sharex="all", figsize=(20, 10))
        print("DOING", dw.ID)
        plot_peaks(*dw.left, thresh=75, ax=tax, title="LEFT")
        plot_peaks(*dw.right, thresh=75, ax=bax, title="RIGHT")
        plt.tight_layout()
        plt.suptitle(dw.ID, horizontalalignment="left")
        plt.show()


if __name__ == '__main__':
    main()

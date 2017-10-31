import os

import numpy as np
from matplotlib import pyplot as plt

from utility.peak import find_peaks
from utility.frame import DataWrapper
from utility.operation import average_filter

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


def annotate_peaks(tpeaks, dpeaks, annot, ax):
    Y = np.linalg.norm(dpeaks, axis=1) if dpeaks.ndim > 1 else tpeaks
    labels = ["clap", "jab", "hook", "ucut"]
    for x, y, a in zip(tpeaks, Y, annot):
        ax.annotate(labels[a], xy=(x, y))


def plot_peaks(time, data, thresh, ax=None, title="", annot=None):
    ax = plt.gca() if ax is None else ax
    Y = np.linalg.norm(data, axis=1) if data.ndim > 1 else data
    peaks = find_peaks(Y, threshold=thresh)
    print("FOUND", len(peaks), "peaks")
    ax.plot(time, Y)
    ax.plot(time[peaks], Y[peaks], "rx")
    ax.plot(time, np.ones_like(time)*thresh, "r--")
    if annot is not None:
        annotate_peaks(time[peaks], Y[peaks], annot, ax=ax)
    ax.set_title(title)
    ax.grid()
    return ax


def plot_peaks_twoway(time, data, thresh, ax=None, title="", annot=None):
    ax = plt.gca() if ax is None else ax
    Y = np.linalg.norm(data, axis=1) if data.ndim > 1 else data
    tpeaks = find_peaks(Y, threshold=thresh)
    bpeaks = find_peaks(-Y, threshold=thresh)
    ax.plot(time, Y)
    ax.plot(time[tpeaks], Y[tpeaks], "rx")
    ax.plot(time[bpeaks], Y[bpeaks], "rx")
    ax.plot(time, np.ones_like(time)*thresh, "r--")
    ax.plot(time, np.ones_like(time)*-thresh, "r--")
    if annot is not None:
        annotate_peaks(time[tpeaks], Y[tpeaks], annot[0], ax=ax)
        annotate_peaks(time[bpeaks], Y[bpeaks], annot[1], ax=ax)
    ax.set_title(title)
    ax.grid()
    return ax


def plot_peaks_subtract(dw, thresh, filtersize=3):
    time, nl = dw.get_data("left", norm=True)
    nr = dw.get_data("right", norm=True)[-1]
    left, right = nl - nr, nr - nl
    if filtersize:
        lY, rY = average_filter(left, filtersize), average_filter(right, filtersize)
        _, (tx, bx) = plt.subplots(2, 1, sharex=True)
        plot_peaks_twoway(time, left, thresh, ax=tx, title="UNFILT")
        plot_peaks_twoway(time, lY, thresh, ax=bx, title=f"FILT ({filtersize})", annot=dw.annot)
    else:
        fleft, fright = left, right
        _, (tx, bx) = plt.subplots(2, 1, sharex=True)
        plot_peaks(time, fleft, thresh, tx, annot=dw.get_annotations("left"), title="LEFT")
        plot_peaks(time, fright, thresh, bx, annot=dw.get_annotations("right"), title="RIGHT")
    plt.suptitle(dw.ID)
    plt.show()


def plot_peaks_fft(time, data):
    Y = np.linalg.norm(data, axis=1)
    X = np.arange(1, len(Y)+1)
    fY = np.fft.fft(Y)
    fig, (tx, mx, bx) = plt.subplots(3, 1)
    tx.plot(X, Y)
    mx.plot(X, fY.real)
    fY[0] = 0j
    fY[1000:6500] = 0j
    iY = np.fft.ifft(fY)
    bx.plot(X, iY)
    plt.show()


def main():
    for file in sorted(os.listdir(logroot)):
        dw = DataWrapper(logroot + file)
        fig, (tax, bax) = plt.subplots(2, 1, sharex="all", figsize=(20, 10))
        print("DOING", dw.ID)
        X = np.arange(len(dw.left[-1]))
        plot_peaks(X, dw.left[-1], thresh=70, ax=tax, title="LEFT")
        plot_peaks(X, dw.right[-1], thresh=70, ax=bax, title="RIGHT")
        plt.tight_layout()
        plt.suptitle(dw.ID, horizontalalignment="left")
        plt.show()


if __name__ == '__main__':
    # for dw in (DataWrapper(logroot + file) for file in os.listdir(logroot)):
    #     plot_peaks_subtract(dw, thresh=50, filtersize=5)
    plot_peaks_subtract(DataWrapper("Bela_fel"), thresh=40, filtersize=5)

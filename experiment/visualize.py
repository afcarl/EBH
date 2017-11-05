import numpy as np
from matplotlib import pyplot as plt

from EBH.utility.peak import find_peaks
from EBH.utility.frame import DataWrapper
from EBH.utility.operation import average_filter
from EBH.utility.const import labels


def plot_acceleration(dw, show=True, dumppath=None):
    ltime, ldata, rtime, rdata = dw.data
    fig, axarr = plt.subplots(2, 2, figsize=(20, 10))
    for i, (lcol, rcol, ax) in enumerate(zip(ldata.T, rdata.T, axarr.flat[:3]), start=1):
        ax.set_title("Axis {}".format(i))
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
    if annot is None:
        return
    Y = np.linalg.norm(dpeaks, axis=1) if dpeaks.ndim > 1 else dpeaks
    for x, y, a in zip(tpeaks, Y, annot):
        top = y > 0
        va = "bottom" if top else "top"
        offs = 2 * 1 if top else -1
        ax.annotate(labels[a], xy=(x, y+offs), verticalalignment=va, horizontalalignment="center")


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


def plot_peaks_twoway(time, data, threshtop, threshbot=None, ax=None, title="", annot=None):
    ax = plt.gca() if ax is None else ax
    threshbot = threshtop if threshbot is None else threshbot
    Y = np.linalg.norm(data, axis=1) if data.ndim > 1 else data
    tpeaks = find_peaks(Y, threshold=threshtop)
    bpeaks = find_peaks(-Y, threshold=threshbot)
    ax.plot(time, Y)
    ax.plot(time[tpeaks], Y[tpeaks], "rx")
    ax.plot(time[bpeaks], Y[bpeaks], "rx")
    ax.plot(time, np.ones_like(time) * threshtop, "r--")
    ax.plot(time, np.ones_like(time) * -threshbot, "r--")
    if annot:
        annotate_peaks(time[tpeaks], Y[tpeaks], annot[0], ax=ax)
        annotate_peaks(time[bpeaks], Y[bpeaks], annot[1], ax=ax)
    ax.set_title(title)
    ax.grid()
    return ax


# noinspection PyTypeChecker
def plot_peaks_subtract(dw, threshtop, threshbot=None, filtersize=3):
    if threshbot is None:
        threshbot = threshtop
    time, nl = dw.get_data("left", norm=True)
    nr = dw.get_data("right", norm=True)[-1]
    left, right = nl - nr, nr - nl
    if filtersize > 1:
        lY, rY = average_filter(left, filtersize), average_filter(right, filtersize)
    else:
        lY, rY = left, right
    _, (tx, bx) = plt.subplots(2, 1, sharex=True)
    plot_peaks_twoway(time, left, threshtop, threshbot, ax=tx, title="UNFILT")
    plot_peaks_twoway(time, lY, threshtop, threshbot, ax=bx, title="FILT ({})".format(filtersize), annot=dw.annot)
    plt.suptitle(dw.ID)
    plt.get_current_fig_manager().window.showMaximized()
    plt.tight_layout()
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


if __name__ == '__main__':
    # for dw in (DataWrapper(logroot + file) for file in os.listdir(logroot)):
    #     plot_peaks_subtract(dw, thresh=50, filtersize=5)
    dw = DataWrapper("box4_fel")
    plot_peaks_subtract(dw,
                        dw.cfg.get("threshtop", 40),
                        dw.cfg.get("threshbot", 40),
                        dw.cfg.get("filtersize", 3))

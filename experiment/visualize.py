import numpy as np
from matplotlib import pyplot as plt

from EBH.utility.peak import find_peaks
from EBH.utility.frame import DataWrapper
from EBH.utility.operation import average_filter
from EBH.utility.const import labels


def plot_acceleration(dw, show=True, dumppath=None):
    ldata, rdata = dw.get_data("left", 0), dw.get_data("right", 0)
    time = dw.time
    fig, axarr = plt.subplots(2, 2, figsize=(20, 10))
    for i, (lcol, rcol, ax) in enumerate(zip(ldata.T, rdata.T, axarr.flat[:3]), start=1):
        ax.set_title("Axis {}".format(i))
        ax.plot(time, lcol)
        ax.plot(time, rcol)
        ax.grid()
    axarr[-1, -1].plot(time, np.linalg.norm(ldata, axis=1), label="Left")
    axarr[-1, -1].plot(time, np.linalg.norm(rdata, axis=1), label="Right")
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


def plot_peaks_twoway(time, data, threshtop, threshbot=None, mindist=10, ax=None, title="", annot=None):
    ax = plt.gca() if ax is None else ax
    threshbot = threshtop if threshbot is None else threshbot
    Y = np.linalg.norm(data, axis=1) if data.ndim > 1 else data
    tpeaks = find_peaks(Y, threshold=threshtop, mindist=mindist)
    bpeaks = find_peaks(-Y, threshold=threshbot, mindist=mindist)
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
def plot_peaks_subtract(dw):
    filtersize, mindist = dw.cfg["filtersize"], dw.cfg["mindist"]
    threshtop, threshbot = dw.cfg["threshtop"], dw.cfg["threshbot"]
    nl = dw.get_data("left", 0, norm=True)
    nr = dw.get_data("right", 0, norm=True)
    left = nl - nr

    lY = average_filter(left, filtersize) if filtersize > 1 else left
    _, (tx, bx) = plt.subplots(2, 1, sharex=True)
    plot_peaks_twoway(dw.time, left, threshtop, threshbot, mindist=mindist, ax=tx,
                      title="UNFILT")
    plot_peaks_twoway(dw.time, lY, threshtop, threshbot, mindist=mindist, ax=bx,
                      title=f"FILT ({filtersize})", annot=(dw.get_annotations("left"),
                                                           dw.get_annotations("right")))
    plt.suptitle(dw.ID)
    plt.get_current_fig_manager().window.showMaximized()
    plt.subplots_adjust(top=0.8, bottom=0.041, left=0.033, right=0.99, hspace=0.093, wspace=0.2)
    plt.show()


def plot_peaks_fft(dw):
    Y = dw.get_data("left", 0, norm=True)
    X = np.arange(1, len(Y)+1)
    fY = np.fft.fft(Y)
    fig, (tx, mx, bx) = plt.subplots(3, 1)
    tx.plot(X, Y)
    mx.plot(X, fY.real)
    fY[0] = 0j
    fY[1000:6500] = 0j
    iY = np.fft.ifft(fY)
    bx.plot(X, iY)


if __name__ == '__main__':
    dwrap = DataWrapper("Szilard_fel", cliptime=False)
    plot_peaks_subtract(dwrap)

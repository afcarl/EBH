import numpy as np
from matplotlib import pyplot as plt

from EBH.utility.frame import DataWrapper
from EBH.utility.operation import as_string

STEP = 15
PEAKSIZE = 20
READINGFRAME = 0

dw = DataWrapper("Virginia_le")
X, Y = dw.get_learning_table(PEAKSIZE)
# X, Y = load_dataset()
Y = as_string(Y)
N, peaksize, axes = X.shape


def meanlet(Xs, Ys, label):
    return Xs[Ys == label].mean(axis=0)


def plot_diff():
    d = (meanlet(X, Y, "H") - meanlet(X, Y, "U"))
    plt.plot(d)
    plt.title(f"$D = {np.sqrt(d.sum()):.2f}$")
    plt.show()


def plot_mean_wavelets(name):
    dwtop = DataWrapper(name + "_fel")
    dwbot = DataWrapper(name + "_le")
    topX, topY = dwtop.get_learning_table(PEAKSIZE, READINGFRAME)
    botX, botY = dwbot.get_learning_table(PEAKSIZE, READINGFRAME)
    topY, botY = map(as_string, (topY, botY))
    meanlets = (list(map(lambda lbl: meanlet(topX, topY, lbl), "JHU")) +
                list(map(lambda lbl: meanlet(botX, botY, lbl), "JHU")))
    # noinspection PyTypeChecker
    fig, axarr = plt.subplots(3, 2, sharey=True, sharex=True, figsize=(20, 10))
    label = ["top" + l for l in "JHU"] + ["bot" + l for l in "JHU"]
    for label, ax, mlet in zip(label, axarr.T.flat, meanlets):
        mlet = np.abs(mlet)
        for n, x in zip("xyz", mlet.T):
            ax.plot(x, label=n)
        ax.plot(np.linalg.norm(mlet, axis=1), "r-", label="L2")
        ax.set_title(f"Mean wavelet for {label}")
        ax.set_xticks(range(0, PEAKSIZE))
        ax.grid()
    plt.legend()
    plt.suptitle(name)
    plt.tight_layout()
    plt.show()


def plot_peaks():
    for start in range(0, len(X), STEP):
        fig, axarr = plt.subplots(5, 3, figsize=(20, 10))
        for peak, annot, ax in zip(X[start:start+STEP], Y[start:start+STEP], axarr.flat):
            ax.plot(peak)
            ax.set_title(f"Annot: {annot}")
            ax.set_xticks(range(0, peaksize))
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    plot_mean_wavelets("Toni")

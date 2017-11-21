import numpy as np
from matplotlib import pyplot as plt

from EBH.utility.operation import load_testsplit_dataset

STEP = 15
sgm = lambda z: 1. / 1. + np.exp(-z)

_, _, X, Y = load_testsplit_dataset("Virginia", as_string=True, mxnormalize=True, y_transform=np.tanh)
N, peaksize, axes = X.shape


def meanlet(Xs, Ys, label):
    return Xs[Ys == label].mean(axis=0)


def plot_diff():
    d = (meanlet(X, Y, "H") - meanlet(X, Y, "U"))[:, 1]**2.
    plt.plot(d)
    plt.title(f"$D = {np.sqrt(d.sum()):.2f}$")
    plt.show()


def plot_mean_wavelets():
    fig, axarr = plt.subplots(3, 1, figsize=(20, 10))
    for label, ax in zip("JHU", axarr):
        wavelet = np.median(X[Y == label], axis=0)
        ax.plot(wavelet)
        ax.set_title(f"Mean wavelet for {label}")
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
    plot_mean_wavelets()

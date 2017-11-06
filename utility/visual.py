import numpy as np
from matplotlib import pyplot as plt


def plot_curve(curveX, curveY, *args, ax=None, show=True, dumppath=None, **pltarg):
    xlab, ylab = pltarg.pop("axlabels", ("", ""))
    title = pltarg.pop("title", "")
    if ax is None:
        ax = plt.gca()
    ax.plot(curveX, curveY, *args, **pltarg)
    if xlab is not None:
        ax.set_xlabel(xlab)
    if ylab is not None:
        ax.set_ylabel(ylab)
    if title is not None:
        ax.set_title(title)
    if dumppath:
        plt.savefig(dumppath)
    if show:
        plt.show()
    return ax


def plot_learning_dynamics(history, show=True, dumppath=None):
    hd = history.history
    epochs = np.arange(1, len(hd["loss"])+1)
    fig, (tax, bax) = plt.subplots(2, 1, sharex=True)
    plot_curve(epochs, hd["loss"], "b-", ax=tax, label="Learning", show=False)
    plot_curve(epochs, hd["val_loss"], "r-", ax=tax, label="Testing", axlabels=("Epochs", "Cost"), show=False)
    plot_curve(epochs, hd["acc"], "b-", ax=bax, label="Learning", show=False)
    plot_curve(epochs, hd["val_acc"], "r-", ax=bax, label="Testing", axlabels=("Epochs", "Accuracy"), show=False)
    tax.legend()
    bax.legend()
    plt.grid()
    plt.tight_layout()
    if dumppath:
        plt.savefig(dumppath)
    if show:
        plt.show()

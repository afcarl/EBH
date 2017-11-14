import numpy as np
from matplotlib import pyplot as plt


def plot_curve(curveX, curveY, *args, ax=None, show=True, dumppath=None, **pltarg):
    xlab, ylab = pltarg.pop("axlabels", ("", ""))
    yticks = pltarg.pop("yticks", None)
    title = pltarg.pop("title", "")
    template = pltarg.pop("template", "")
    va = pltarg.pop("va", "")
    if ax is None:
        ax = plt.gca()
    ax.plot(curveX, curveY, *args, **pltarg)
    if template:
        x, y = curveX[-1], curveY[-1]
        ax.annotate(template.format(y=curveY[-1]), xy=(x, y), verticalalignment=va)
    if xlab is not None:
        ax.set_xlabel(xlab)
    if ylab is not None:
        ax.set_ylabel(ylab)
    if yticks is not None:
        ax.set_yticks(yticks)
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
    ctmp, atmp = "{y:.4f}", "{y:.2%}"
    plot_curve(epochs, hd["loss"], "b-", ax=tax, label="Learning", show=False, template=ctmp, va="top")
    plot_curve(
        epochs, hd["val_loss"], "r-", ax=tax, label="Testing", axlabels=("Epochs", "Cost"),
        show=False, template=ctmp, va="bottom"
    )
    plot_curve(epochs, hd["acc"], "b-", ax=bax, label="Learning", show=False, yticks=[0., 1.],
               template=atmp, va="bottom")
    plot_curve(
        epochs, hd["val_acc"], "r-", ax=bax, label="Testing", axlabels=("Epochs", "Accuracy"),
        show=False, yticks=[0., 1.], template=atmp, va="top"
    )
    tax.legend()
    bax.legend()
    plt.grid()
    plt.tight_layout()
    if dumppath:
        plt.savefig(dumppath)
    if show:
        plt.show()

import numpy as np
from matplotlib import pyplot as plt

from EBH.utility.frame import DataWrapper
from EBH.experiment.plottransform import load_data


def as_angle(stream, against=(0., 1., 0.)):
    other = np.array(against)
    nstream = stream / np.linalg.norm(stream, axis=-1, keepdims=True)
    cosines = nstream.dot(other)
    return np.degrees(np.arccos(cosines))


def plotangles(stream, title):
    angles = map(lambda other: as_angle(stream, other), np.eye(3))
    fig, axes = plt.subplots(3, 1, sharex=True, sharey=True)
    for line, ax in zip(angles, axes):
        ax.plot(line)
        ax.grid(True)
    plt.suptitle(title)
    plt.show()


def peak_mean_angles():
    pass


def xperiment_angleplot():
    dw = DataWrapper("box4_fel")
    plotangles(dw.get_data("left"), "left")
    plotangles(dw.get_data("right"), "right")


def xperiment_meanwavelets():
    X, gesture, name, orient, ID, pro, hand = load_data(False)

    N = len(X)
    X = X.reshape(N, 20, 4).transpose(0, 2, 1).reshape(N, 80)

    left, right = X[hand == "left"], X[hand == "right"]

    left_mean, right_mean = left.mean(axis=0), right.mean(axis=0)
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)

    ((tax, mtx), (btx, bax)) = axes

    tax.plot(left_mean)
    tax.set_title("left")
    mtx.plot(right_mean)
    mtx.set_title("right")
    btx.plot(left_mean - right_mean)
    btx.plot([0. for _ in range(len(left_mean))], "r-")
    btx.set_title("diff")
    bax.plot(X.mean(axis=0))
    bax.set_title("mean")

    for ax in axes.flat:
        ax.grid(True)

    plt.show()

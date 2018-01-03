import numpy as np
from matplotlib import pyplot as plt

from EBH.utility.frame import DataWrapper
from EBH.utility.operation import load_dataset


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


def main():
    dw = DataWrapper("box4_fel")
    plotangles(dw.get_data("left"), "left")
    plotangles(dw.get_data("right"), "right")


if __name__ == '__main__':
    main()

import os
import gzip
import pickle

import numpy as np

from EBH import projectroot, logroot
from utility.frame import DataWrapper


def pull_dws():
    return [dw for dw in (DataWrapper(logroot + lofgl) for lofgl in os.listdir(logroot))
            if dw.annot[0] is not None]


def merge_dws(dws):
    Xs, Ys = [], []
    for dw in dws:  # type: DataWrapper
        print("Extracting", dw.ID)
        topX, botX = dw.get_peaks(peaksize=10, args=False)
        topY, botY = dw.get_annotations("left"), dw.get_annotations("right")
        x, y = np.concatenate((topX, botX)), np.concatenate((topY, botY))
        print("x:", x.shape)
        print("y:", y.shape)
        Xs.append(x)
        Ys.append(y)
    X, Y = np.concatenate(Xs), np.concatenate(Ys)
    print("Extracted X:", X.shape)
    print("Extracted Y:", Y.shape)
    with gzip.open(projectroot + "data.pkl.gz", "wb") as handle:
        pickle.dump((X, Y), handle)


if __name__ == '__main__':
    merge_dws(pull_dws())

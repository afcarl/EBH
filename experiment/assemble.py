import os
import gzip
import pickle

import numpy as np

from EBH.utility.const import logroot, projectroot
from EBH.utility.frame import DataWrapper


def pull_dws():
    out = []
    for logfl in os.listdir(logroot):
        dw = DataWrapper(logroot + logfl)
        out.append(dw)
    return [dw for dw in (DataWrapper(logroot + lofgl) for lofgl in os.listdir(logroot))
            if dw.annot[0] is not None]


def merge_dws(dws):
    Xs, Ys = [], []
    for dw in dws:  # type: DataWrapper
        print("Extracting", dw.ID)
        x, y = dw.get_learning_table(peaksize=10)
        Xs.append(x)
        Ys.append(y)
    X, Y = np.concatenate(Xs), np.concatenate(Ys)
    valid = np.where(Y != 2)
    X, Y = X[valid], Y[valid]
    print("Extracted X:", X.shape)
    print("Extracted Y:", Y.shape)
    with gzip.open(projectroot + "data.pkl.gz", "wb") as handle:
        pickle.dump((X, Y), handle)


if __name__ == '__main__':
    merge_dws(pull_dws())

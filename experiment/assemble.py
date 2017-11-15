import os
import gzip
import pickle

import numpy as np

from EBH.utility.const import logroot, projectroot, labels
from EBH.utility.frame import DataWrapper


def pull_dws():
    out = []
    for logfl in os.listdir(logroot):
        dw = DataWrapper(logroot + logfl, cliptime=True)
        print("Reading", dw.ID, end="... ")
        if dw.annot["l"] is None or dw.annot["r"] is None:
            print("unlabeled!")
            continue
        print("labeled.")
        out.append(dw)
    return out


def merge_dws(dws, include="JHU"):
    Xs, Ys = [], []
    for dw in dws:  # type: DataWrapper
        print("Extracting", dw.ID)
        x, y = dw.get_learning_table(peaksize=10)
        Xs.append(x)
        Ys.append(y)
    X, Y = np.concatenate(Xs), np.concatenate(Ys)
    valid = np.zeros_like(Y, dtype=bool)
    for i in (labels.index(l) for l in include):
        valid |= Y == i
    valid = np.where(valid)
    X, Y = X[valid], Y[valid]
    print("Extracted X:", X.shape)
    print("Extracted Y:", Y.shape)
    with gzip.open(projectroot + "data.pkl.gz", "wb") as handle:
        pickle.dump((X, Y), handle)


if __name__ == '__main__':
    merge_dws(pull_dws())

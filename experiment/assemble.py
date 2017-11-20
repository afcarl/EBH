import os
import gzip
import pickle

import numpy as np

from EBH.utility.const import logroot, labels, ltbroot, DEFAULT_DATASET, boxer_names
from EBH.utility.frame import DataWrapper


def dwstream(exclude=(), include_only="all"):
    files = os.listdir(logroot)
    included = files if include_only == "all" else [f + ".txt" for f in include_only]
    for logfl in included:
        dw = DataWrapper(logroot + logfl, cliptime=True)
        if not dw.is_annotated:
            print("Skipping unlabelled data:", dw.ID)
            continue
        if dw.ID in exclude:
            print("Excluded", dw.ID)
            continue
        yield dw


def assemble_data(dws=None, mergehplane=False, augment=False, peaksize=10, includeclass="JHU"):
    dws = dwstream() if dws is None else dws
    Xs, Ys = [], []
    for dw in dws:  # type: DataWrapper
        print("Extracting", dw.ID)
        y = np.r_[dw.get_annotations()]
        if augment:
            x = np.concatenate([
                np.concatenate(dw.get_peaks(peaksize, readingframe=f, center=False)) for f in range(5)
            ])
            y = np.repeat(np.r_[dw.get_annotations()], 5)
        else:
            x = np.r_[dw.get_peaks(peaksize, readingframe=0, center=False)]
        if len(x) != len(y):
            print("Possibly unfinished annotation (x/y lengths differ). Skipping!")
            continue
        Xs.append(x)
        Ys.append(y)
    X, Y = np.concatenate(Xs), np.concatenate(Ys)
    if mergehplane:
        y = X[:, :, 1]
        p = np.linalg.norm(X, axis=2)
        if augment:
            y = np.concatenate((y, -y))
            p = np.concatenate((p, p))
            Y = np.concatenate((Y, Y))
        X = np.concatenate((p, y), axis=-1)
    return filter_classes(X, Y, include=includeclass)


def filter_classes(X, Y, include):
    valid = np.zeros_like(Y, dtype=bool)
    for i in (labels.index(l) for l in include):
        valid |= Y == i
    valid = np.where(valid)
    return X[valid], Y[valid]


def _dump_dataset(X, Y, path=DEFAULT_DATASET):
    with gzip.open(path, "wb") as handle:
        pickle.dump((X, Y), handle)
    print("Dumped dataset to", path)


def merge_dws(include="JHU", mergehplane=False, augment=False):
    X, Y = assemble_data(dwstream(), mergehplane, augment, includeclass=include)
    print("Final extracted X:", X.shape)
    print("Final extracted Y:", Y.shape)
    _dump_dataset(X, Y)


def separate_sessions(separated=(), include_labels="JHU", mergehplane=False, augment=False):
    lX, lY = assemble_data(
        dwstream(exclude=separated), mergehplane=mergehplane, augment=augment, includeclass=include_labels
    )
    tX, tY = assemble_data(
        dwstream(include_only=separated), mergehplane=mergehplane, augment=augment, includeclass=include_labels
    )
    print("Assembled learning X:", lX.shape)
    print("Assembled learning Y:", lY.shape)
    print("Assembled test X:", tX.shape)
    print("Assembled test Y:", tY.shape)
    print("Spit for testing:", " ".join(separated))
    _dump_dataset(lX, lY, path=ltbroot+"learning.pkl.gz")
    _dump_dataset(tX, tY, path=ltbroot+"testing.pkl.gz")
    with open(ltbroot + "splitset.meta.txt", "w") as handle:
        handle.write("Split for testing:\n")
        handle.write("\n".join(separated))
    return lX, lY, tX, tY


def build_leave_one_out_datasets(include_labels="JHU", mergehplane=False, augment=False):
    dws = list(dwstream())
    N = len(boxer_names)
    for i, name in enumerate(boxer_names, start=1):
        print(f"Building dataset - excluded: {name} ({i}/{N})")
        lX, lY = filter_classes(*assemble_data(
            (dw for dw in dws if name not in dw.ID), mergehplane=mergehplane, augment=augment), include=include_labels)
        _dump_dataset(lX, lY, path=ltbroot + f"E_{name}_learning.pkl.gz")
        tX, tY = filter_classes(*assemble_data(
            (dw for dw in dws if name in dw.ID), mergehplane=mergehplane, augment=augment), include=include_labels)
        _dump_dataset(tX, tY, path=ltbroot + f"E_{name}_testing.pkl.gz")


if __name__ == '__main__':
    build_leave_one_out_datasets(mergehplane=False, augment=False)
    # merge_dws()

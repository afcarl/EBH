import os
import gzip
import pickle

import numpy as np

from .parse import extract_data, pull_annotation
from .peak import find_peaks, find_peaks_subtract
from EBH import logroot, labroot


class DataWrapper:

    def __init__(self, source):
        if ".txt" == source[-4:]:
            data = extract_data(source)
            self.ID = os.path.split(source)[-1].split(".")[0]
        elif ".wrp" == source[-4:]:
            self.ID, data = DataWrapper.load(source)
        else:
            # Assume source is ID
            self.ID = source
            data = extract_data(logroot + f"{source}.txt")
        labpath = f"{labroot}{self.ID}.txt"
        if os.path.exists(labpath):
            a = pull_annotation(labpath)
            self.annot = {"l": a[0], "r": a[1], 0: a[0], 1: a[1]}
        else:
            self.annot = None
        self.data = {"l": data[:2], "r": data[2:]}

    def get_data(self, side=None, norm=False):
        if side is None:
            dset = (np.concatenate((self.data["l"][0], self.data["r"][0])),
                    np.concatenate((self.data["l"][1], self.data["r"][1])))
        else:
            dset = self.data[str(side)[0].lower()]
        if norm:
            return dset[0], np.linalg.norm(dset[1], axis=1)
        return dset

    def get_peaks_vanilla(self, side=None, threshold=75, peaksize=10, appendnorm=True):
        time, data = self.get_data(side)
        norm = np.linalg.norm(data, axis=1)
        peakarg = find_peaks(norm, threshold=threshold, center=True)
        hsz = peaksize // 2
        data = np.concatenate((data, norm[:, None]), axis=-1) if appendnorm else data
        X = np.array([data[p-hsz:p+hsz] for p in peakarg])
        print("X size:", X.shape)
        return X

    def get_peaks_subtract(self, threshold=50, peaksize=10, appendnorm=True):
        time, left = self.get_data("left")
        right = self.get_data("right")[-1]
        if appendnorm:
            ldata = np.concatenate((left, np.linalg.norm(left, axis=1, keepdims=True)), axis=-1)
            rdata = np.concatenate((right, np.linalg.norm(right, axis=1, keepdims=True)), axis=-1)
        else:
            ldata = left
            rdata = right
        top, bot = find_peaks_subtract(self, threshold, center=True)
        hsz = peaksize // 2
        topX = np.array([ldata[p-hsz:p+hsz] for p in top if len(ldata)-hsz-1 > p > hsz])
        botX = np.array([rdata[p-hsz:p+hsz] for p in bot if len(rdata)-hsz-1 > p > hsz])
        return topX, botX

    def get_annotations(self, side=None):
        if self.annot is None:
            return
        return {"N": np.concatenate(self.annot), "l": self.annot[0], "r": self.annot[1]}[str(side)[0]]

    @staticmethod
    def load(source):
        return pickle.load(gzip.open(source))

    def save(self, dest):
        with gzip.open(dest) as handle:
            pickle.dump(self, handle)

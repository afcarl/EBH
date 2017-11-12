import os
import gzip
import pickle

import numpy as np

from .peak import find_peaks_subtract
from .const import logroot, labroot, pklroot
from .parse import extract_data, pull_annotation


class DataWrapper:

    def __init__(self, source):
        if ".txt" == source[-4:]:
            self.ID = os.path.split(source)[-1].split(".")[0]
            data = extract_data(source)
        else:
            self.ID = source
            data = extract_data(logroot + "{}.txt".format(source))
        labpath = "{}{}.txt".format(labroot, self.ID)
        if os.path.exists(labpath):
            a = pull_annotation(labpath)
            print("Found labels with config:", a[-1])
        else:
            a = [None, None, {}]
        self.annot = {"l": a[0], "r": a[1], 0: a[0], 1: a[1]}
        self.data = {"l": data[:2], "r": data[2:]}
        self.annotated = {"l": [], "r": []}
        self.cfg = a[2]

    def adjust_threshold(self):
        if self.annot["l"] is None or self.annot["r"] is None:
            return
        lN = len(self.annot["l"])
        rN = len(self.annot["r"])
        self.cfg["threshtop"] = 40
        self.cfg["threshbot"] = 40
        self.cfg["filtersize"] = 3
        for i in range(40):
            lpeaks, rpeaks = self.get_peaks(peaksize=10, args=True)
            lgood = len(lpeaks) >= lN
            rgood = len(rpeaks) >= rN
            if rgood and lgood:
                break
            if not lgood and self.cfg["threshtop"] > 20:
                self.cfg["threshtop"] -= 1
            if not rgood and self.cfg["threshbot"] > 20:
                self.cfg["threshbot"] -= 1
        else:
            print("Couldn't adjust threshold for", self.ID)
        print("Adjusted config to", self.cfg)

    def get_data(self, side=None, norm=False):
        if side is None:
            dset = (np.concatenate((self.data["l"][0], self.data["r"][0])),
                    np.concatenate((self.data["l"][1], self.data["r"][1])))
        else:
            dset = self.data[str(side)[0].lower()]
        if norm:
            return dset[0], np.linalg.norm(dset[1], axis=1)
        return dset

    def get_peaks(self, peaksize=10, args=True):
        top, bot = find_peaks_subtract(self, threshtop=self.cfg.get("threshtop", 35),
                                       threshbot=self.cfg.get("threshbot", 35),
                                       filtersize=self.cfg.get("filtersize", 3),
                                       peaksize=0)
        if args:
            return top, bot
        hsz = peaksize // 2
        time, left = self.get_data("left")
        right = self.get_data("right")[-1]
        lefttop = len(left)-peaksize-1
        righttop = len(right)-peaksize-1
        topX = np.array([left[p-hsz:p+hsz] for p in top if peaksize < p < lefttop])
        botX = np.array([right[p-hsz:p+hsz] for p in bot if peaksize < p < righttop])
        return topX, botX

    def get_annotations(self, side=None):
        if self.annot is None:
            return
        return {"N": np.concatenate((self.annot["l"], self.annot["r"])),
                "l": self.annot[0], "r": self.annot[1]}[str(side)[0]]

    def get_learning_table(self, peaksize=10):
        X = np.concatenate(self.get_peaks(peaksize, args=False))
        Y = self.get_annotations(side=None)
        assert len(X) == len(Y), f"Lengths not equal in {self.ID}: X: {X.shape} Y: {Y.shape}"
        return X, Y

    @staticmethod
    def load(ID):
        return pickle.load(gzip.open(pklroot + ID))

    def save(self):
        with gzip.open(pklroot + self.ID) as handle:
            pickle.dump(self, handle)

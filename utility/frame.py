import os

import numpy as np

from .peak import find_peaks_subtract
from .const import logroot, labroot
from .parse import extract_data, extract_data_raw, pull_annotation


class DataWrapper:

    def __init__(self, source, cliptime=True, extract_mode="zolaly"):
        self.cfg = dict(threshtop=40, threshbot=40, filtersize=3, mindist=10)
        if ".txt" == source[-4:]:
            self.ID = os.path.split(source)[-1].split(".")[0]
            data = extract_data(source)
        else:
            self.ID = source
            data = {"zolaly": extract_data, "raw": extract_data_raw
                    }[extract_mode](f"{logroot}{source}.txt", clip=cliptime)
        labpath = f"{labroot}{self.ID}.txt"
        a = pull_annotation(labpath) if os.path.exists(labpath) else [None, None, {}]
        self.annot = {"l": a[0], "r": a[1]}
        self.data = {"l": data[:2], "r": data[2:]}
        self.cfg.update(a[2])

    @property
    def is_annotated(self):
        return not (self.annot["l"] is None or self.annot["r"] is None)

    def adjust_threshold(self):
        if not self.is_annotated:
            return
        lN = len(self.annot["l"])
        rN = len(self.annot["r"])
        self.cfg["threshtop"] = 40
        self.cfg["threshbot"] = 40
        self.cfg["filtersize"] = 3
        self.cfg["mindist"] = 10
        for i in range(40):
            lpeaks, rpeaks = self.get_peaks(peaksize=10, center=True)
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

    def get_peaks(self, peaksize=10, center=True):
        top, bot = find_peaks_subtract(
            self, threshtop=self.cfg["threshtop"], threshbot=self.cfg["threshbot"],
            filtersize=self.cfg["filtersize"], mindist=self.cfg["mindist"],
            peaksize=peaksize
        )
        if center:
            return top, bot
        (ltime, ldata), (rtime, rdata) = self.data["l"], self.data["r"]
        hsize = peaksize // 2
        topX = np.array([ldata[p-hsize:p+hsize] for p in top])
        botX = np.array([rdata[p-hsize:p+hsize] for p in bot])
        return topX, botX

    def get_annotations(self, side=None):
        if not self.is_annotated:
            print("Incomplete annotation in", self.ID)
            return None
        return self.annot.get(str(side).lower()[0], np.r_[self.annot["l"], self.annot["r"]])

    def get_learning_table(self, peaksize=10):
        X = np.concatenate(self.get_peaks(peaksize, center=False))
        Y = self.get_annotations(side=None)
        assert len(X) == len(Y), f"Lengths not equal in {self.ID}: X: {X.shape} Y: {Y.shape}"
        return X, Y

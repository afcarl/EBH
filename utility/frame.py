import os

import numpy as np

from .peak import find_peaks_subtract
from .const import logroot, labroot
from .parse import extract_data, pull_annotation


class DataWrapper:

    def __init__(self, source, cliptime=True):
        self.cfg = dict(threshtop=40, threshbot=40, filtersize=3, mindist=10)
        if ".txt" == source[-4:]:
            ID = os.path.split(source)[-1].split(".")[0]
        else:
            ID = source
            source = f"{logroot}{source}.txt"
        self.boxer, self.orientation = ID.split("_")
        # Data shape: [hand, N, frame, axis]
        self.time, data = extract_data(source, clip=cliptime)
        labpath = f"{labroot}{self.ID}.txt"
        atop, abot, acfg = pull_annotation(labpath) if os.path.exists(labpath) else [None, None, {}]
        topleft = acfg.pop("left", False)
        annot = [atop, abot]
        self._data = {"l": data[int(topleft)], "r": data[int(~topleft)]}
        self._annot = {"l": annot[int(topleft)], "r": annot[int(~topleft)]}
        self.cfg.update(acfg)

    @property
    def ID(self):
        return "_".join((self.boxer, self.orientation))

    @property
    def is_annotated(self):
        return not (self._annot["l"] is None or self._annot["r"] is None)

    def adjust_threshold(self):
        if not self.is_annotated:
            return
        lN = len(self._annot["l"])
        rN = len(self._annot["r"])
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

    def get_data(self, side, readingframe=0, norm=False):
        dset = self._data[side[0]][:, readingframe]
        if norm:
            return np.linalg.norm(dset, axis=1)
        return dset

    def get_peaks(self, peaksize=10, readingframe=0, center=True):
        top, bot = find_peaks_subtract(
            self, threshtop=self.cfg["threshtop"], threshbot=self.cfg["threshbot"],
            filtersize=self.cfg["filtersize"], mindist=self.cfg["mindist"],
            peaksize=peaksize
        )
        if center:
            return top, bot
        ldata, rdata = self.get_data("left", readingframe), self.get_data("right", readingframe)
        hsize = peaksize // 2
        topX = np.array([ldata[p-hsize:p+hsize] for p in top])
        botX = np.array([rdata[p-hsize:p+hsize] for p in bot])
        return topX, botX

    def get_annotations(self, side=None):
        if not self.is_annotated:
            print("Incomplete annotation in", self.ID)
            return None
        if side is None:
            return self._annot["l"], self._annot["r"]
        return self._annot.get(str(side).lower()[0])

    def get_learning_table(self, peaksize=20, readingframe=0, splitsides=False):

        def dropboundary(pt, pb, at, ab, phs, mxln):
            tnope = np.logical_or(pt-phs < 0, pt+phs > mxln)
            bnope = np.logical_or(pb-phs < 0, pb+phs > mxln)
            return pt[~tnope], at[~tnope], pb[~bnope], ab[~bnope]

        toparg, botarg = find_peaks_subtract(
            self, threshtop=self.cfg["threshtop"], threshbot=self.cfg["threshbot"],
            filtersize=self.cfg["filtersize"], mindist=self.cfg["mindist"],
            peaksize=10
        )
        hsize = peaksize // 2
        topY, botY = self.get_annotations(side=None)
        top, bot = self.get_data("left", readingframe), self.get_data("right", readingframe)
        assert len(topY) == len(toparg) and len(botY) == len(botarg), \
            f"topY: {len(topY)} v top: {len(toparg)}; botY: {len(botY)} v bot: {len(botarg)}"

        toparg, topY, botarg, botY = dropboundary(toparg, botarg, topY, botY, peaksize//2, len(top)-1)
        topX = np.array([top[p-hsize:p+hsize] for p in toparg])
        botX = np.array([bot[p-hsize:p+hsize] for p in botarg])
        assert len(topX) == len(topY)
        assert len(botX) == len(botY)
        return (topX, topY, botX, botY) if splitsides else (np.r_[topX, botX], np.r_[topY, botY])

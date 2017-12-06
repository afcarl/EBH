from collections import deque, namedtuple
from types import SimpleNamespace


GRData = namedtuple("GRData", "ID", "activity", "hand", "event", "intensity", "angle", "gc", "fwd")

GRRawHandleData = namedtuple("GRRawHandleData", "deviceId", "message_cntr", "relative_cntr",
                             "status_byte", "acc_x", "acc_y", "acc_z")


class GRStats(SimpleNamespace):

    params = ("nLJab", "nLJabU", "nLHook", "nLHookU", "nLUCut", "nLUCutU",
              "nRJab", "nRJabU", "nRHook", "nRHookU", "nRUCut", "nRUCutU")

    def __init__(self, **kw):
        for p in self.params:
            if p not in kw:
                kw[p] = 0
        super().__init__(**kw)

    @property
    def nLHits(self):
        return self.nLJab + self.nLHook + self.nLUCut

    @property
    def nLHitsU(self):
        return self.nLJabU + self.nLHookU + self.nLUCutU

    @property
    def nLHitsTotal(self):
        return self.nLHits + self.nLHitsU

    @property
    def nRHits(self):
        return self.nRJab + self.nRHook + self.nRUCut

    @property
    def nRHitsU(self):
        return self.nRJabU + self.nRHookU + self.nRUCutU

    @property
    def nRHitsTotal(self):
        return self.nRHits + self.nRHitsU

    @property
    def nHits(self):
        return self.nLHits + self.nRHits

    @property
    def nHitsU(self):
        return self.nLHitsU + self.nRHitsU

    @property
    def nHitsTotal(self):
        return self.nLHitsTotal + self.nRHitsTotal


class GestureRecognizer:

    def __init__(self, threshold, peaksize):

        self.model = None
        self.threshold = threshold
        self.memory = [0 for _ in range(peaksize)]

    def feed(self, rawdata: GRRawHandleData):
        pass

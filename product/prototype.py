from collections import deque, namedtuple
from types import SimpleNamespace


GRData = namedtuple("GRData", "ID", "activity", "hand", "event", "intensity", "angle", "gc", "fwd")


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
        self.memory = deque(maxlen=10)

    def feed(self, left, right, result):
        pass

    def _translate_hexa(self):
        pass

    def fit_model(self, X, Y):
        pass

    def incorporate(self, hexa):
        pass

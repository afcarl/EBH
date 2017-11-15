from collections import deque


class GestureRecognizer:

    def __init__(self, threshold, peaksize):

        self.model = None
        self.threshold = threshold
        self.memory = deque(maxlen=10)

    def _translate_hexa(self):
        pass

    def fit_model(self, X, Y):
        pass

    def incorporate(self, hexa):
        pass

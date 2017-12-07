import numpy as np

from EBH.utility.const import projectroot


class KNN:

    def __init__(self):
        raw = open(projectroot + "neighborhood.bin", "rb").read()
        self.N = np.fromstring(raw[:2], dtype="int16")[0]
        self.Y = np.fromstring(raw[2:self.N+2], dtype="int8").astype(int)
        self.X = np.fromstring(raw[self.N+2:], dtype="int8").astype(int)
        self.X = self.X.reshape(self.N, len(self.X) // self.N)

    def predict(self, x):
        d = np.sum(np.abs(self.X - x), axis=1)
        assert len(d) == len(self.X)
        votes = [0, 0, 0]
        for arg in np.argsort(d)[:5]:
            votes[self.Y[arg]] += 1
        return np.argmax(votes)

    def evaluate(self):
        preds = []
        for i, x in enumerate(self.X, start=1):
            print(f"\r{len(self.X)}/{i}", end="")
            preds.append(self.predict(x))
        print()
        print("Current accuracy:", (preds == self.Y).mean())


if __name__ == '__main__':
    KNN().evaluate()

import numpy as np


def _asfixed(num, dtype=int):
    return np.array([num], dtype=dtype).tobytes()


class DataConfig:

    DEPTH = None

    def __init__(self, X, Y, more_labels):
        self.X, self.Y = X, Y
        self.N, self.PEAK = X.shape[:2]
        self.labels = more_labels

    def apply(self): raise NotImplementedError

    @property
    def PEAKSIZE(self):
        return self.PEAK * self.DEPTH

    def _generate_generic_info(self):
            return b"".join(map(_asfixed, [self.N, self.PEAK, self.DEPTH]))

    def _generate_neighborhood(self):
        Ys = self.Y.astype("uint8").tobytes()
        Xs = self.X.astype("uint8").tobytes()
        return Ys + Xs

    def _generate_pspace(self):

        def calc_class(cls):
            Xc = self.X[self.Y == cls]
            mu_bin = Xc.mean(axis=0).astype("float32").tobytes()
            isigma_bin = np.linalg.inv(np.cov(Xc.T)).astype("float32").tobytes()
            return mu_bin, isigma_bin

        mus, icovs = zip(*map(calc_class, "JHU"))
        return b"".join(mus) + b"".join(icovs)

    def generate_config_files(self, output_root):
        with open(output_root + "generic.bin", "wb") as handle:
            handle.write(self._generate_generic_info())
        with open(output_root + "neigborhood.bin", "wb") as handle:
            handle.write(self._generate_neighborhood())
        with open(output_root + "pspace.bin", "wb") as handle:
            handle.write(self._generate_pspace())


class MergeXZ(DataConfig):

    DEPTH = 2

    def apply(self):
        x, y, z = self.X.T
        return np.stack(((x + z).T / 2., np.abs(y.T)), axis=-1)


class ExtendWithAbsY(DataConfig):

    DEPTH = 4

    def apply(self):
        x, y, z = self.X.T
        return np.stack((x.T, y.T, np.abs(y.T), z.T), axis=-1)

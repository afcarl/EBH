import numpy as np


def _asfixed(num, dtype="int16"):
    return np.array([num], dtype=dtype).tobytes()


def generate_generic_file(X, output_root):
    N, PEAK, CHAN = map(_asfixed, X.shape)
    with open(output_root + "generic.bin", "wb") as handle:
        handle.write(N + PEAK + CHAN)


def generate_neighborhood_file(X, Y, output_root):
    Ys = Y.astype("uint8").tobytes()
    Xs = X.astype("uint8").tobytes()
    with open(output_root + "neighborhood.bin", "wb") as handle:
        handle.write(Ys + Xs)


def generate_pspace_file(X, Y, output_root):

    def calc_class(cls):
        Xc = X[Y == cls]
        mu_bin = Xc.mean(axis=0).astype("float32").tobytes()
        isigma_bin = np.linalg.inv(np.cov(Xc.T)).astype("float32").tobytes()
        return mu_bin, isigma_bin

    mus, icovs = zip(*map(calc_class, "JHU"))
    with open(output_root + "pspace.bin", "wb") as handle:
        handle.write(b"".join(mus) + b"".join(icovs))


def generate_config_files(X, Y, output_root):
    generate_generic_file(X, output_root)
    generate_neighborhood_file(X, Y, output_root)
    generate_pspace_file(X, Y, output_root)


def merge_x_z(X):
    x, y, z = X.T
    return np.stack(((x + z).T / 2., np.abs(y.T)), axis=-1)


def extend_with_abs_y(X):
    x, y, z = X.T
    return np.stack((x.T, y.T, np.abs(y.T), z.T), axis=-1)


def replace_y_with_abs_y(X):
    x, y, z = X.T
    return np.stack((x.T, np.abs(y.T), z.T), axis=-1)


def extend_with_1norm(X):
    x, y, z = X.T
    return np.stack((x, y, z, (x + y + z).T / 3.), axis=-1)


select = {0: extend_with_abs_y, 1: replace_y_with_abs_y,
          2: extend_with_1norm, 3: merge_x_z}

import numpy as np


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

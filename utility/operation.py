import gzip
import pickle

import numpy as np

from EBH.utility.const import projectroot, labels, onehot


def interpolate_nans(x):
    out = x.copy()
    for i, col in enumerate(x.T):
        nanmask = np.isnan(col)
        nanarg = np.where(nanmask)
        okarg = np.where(~nanmask)
        interp = np.interp(nanarg[0], okarg[0], col[~nanmask])
        out[nanarg, i] = interp
    return out


def average_filter(series, window=2):
    return np.convolve(series, np.ones(window) / window, mode="same")


def shuffle(X, Y, *more):
    arg = np.arange(len(X))
    np.random.shuffle(arg)
    return tuple(array[arg] for array in (X, Y) + more)


def as_onehot(Y, categ=None):
    categ = onehot if categ is None else categ
    return np.array([categ[ix] for ix in Y])


def as_string(Y, lbls=None):
    categ = labels if lbls is None else lbls
    return np.vectorize(lambda ix: categ[ix])(Y)


def as_matrix(X):
    if X.ndim == 2:
        return X
    if X.ndim == 1:
        return X[:, None]
    if X.ndim > 2:
        return X.reshape(X.shape[0], np.prod(X.shape[1:], dtype=int))


def load_dataset(split=0., **kw):
    X, Y = pickle.load(gzip.open(projectroot + "data.pkl.gz"))
    print("Loaded dataset! Shape:", X.shape, end=" ")
    print("Labels:", set(Y))
    if kw.get("as_matrix"):
        X = as_matrix(X)
    if kw.get("normalize"):
        X = normalize(X)
    if kw.get("as_string"):
        Y = as_string(Y, kw.get("labels", labels))
    elif kw.get("as_onehot"):
        Y = as_onehot(Y, kw.get("categ", onehot))
    if split:
        arg = np.arange(len(X))
        np.random.shuffle(arg)
        n = int(len(X)*split)
        larg, targ = arg[n:], arg[:n]
        return X[larg], Y[larg], X[targ], Y[targ]
    return X, Y


def normalize(X, mean=None, std=None, getparam=False):
    mean = X.mean(axis=0, keepdims=True) if mean is None else mean
    std = X.std(axis=0, keepdims=True) if std is None else std
    nX = (X - mean) / std
    return (nX, mean, std) if getparam else nX


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    x = np.linspace(-5, 5, 100)
    X = np.stack((x, x), axis=1)
    X[(10, 20, 30), 0] = np.nan
    X[(0, 1, 2, 3), 1] = np.nan
    Y = np.stack([np.sin(X[:, 0]), np.cos(X[:, 1])], axis=1)
    iY = interpolate_nans(Y)
    plt.plot(Y, "b-", alpha=0.25)
    plt.plot(iY, "rx", alpha=0.5)
    plt.show()

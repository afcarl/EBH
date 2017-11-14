import gzip
import pickle

import numpy as np

from .const import projectroot, labels, onehot


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
    print("Loaded dataset with labels:", set(Y))
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

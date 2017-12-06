import gzip
import pickle

import numpy as np
from EBH.utility.const import labels, DEFAULT_DATASET
from EBH.utility.const import ltbroot


def average_filter(series, window=2):
    return np.convolve(series, np.ones(window) / window, mode="same")


def mahal(x, mu, sigma):
    z = mu - x
    return z @ np.linalg.inv(sigma) @ z


def shuffle(X, Y, *more):
    arg = np.arange(len(X))
    np.random.shuffle(arg)
    return tuple(array[arg] for array in (X, Y) + more)


def split_data(X, Y, alpha, shuff=True):
    arg = np.arange(len(X))
    if shuff:
        np.random.shuffle(arg)
    n = int(len(X) * alpha)
    larg, targ = arg[n:], arg[:n]
    return X[larg], Y[larg], X[targ], Y[targ]


def as_onehot(Y, categ=None):
    categ = np.unique(Y) if categ is None else categ
    onehot = np.eye(len(categ))
    return np.array([onehot[ix] for ix in Y])


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


def drop_category(X, Y, categ, m):
    arg, = np.where(Y == categ)
    m = len(arg) if m is True else m
    np.random.shuffle(arg)
    drops = arg[:m]
    mask = np.ones(len(X), dtype=bool)
    mask[drops] = False
    return X[mask], Y[mask]


def drop_outliers(X, Y, zval=0.95):
    assert Y.ndim == 1
    categ = np.unique(Y)
    newX, newY = [], []
    for cat in categ:
        mask = Y == cat
        Xc = as_matrix(X[mask])
        mu, cov = np.mean(Xc, axis=0), np.cov(Xc.T)
        d = np.apply_along_axis(mahal, 1, Xc, **dict(mu=mu, sigma=cov))
        validarg = np.argsort(d)[:int(zval*len(Xc))]
        newX.append(X[mask][validarg])
        newY.append(Y[mask][validarg])
    return np.concatenate(newX), np.concatenate(newY)


def optimalish_config(learning, testing=None):

    def doit(dset):
        x, y, z = np.split(dset[0], 3, axis=-1)
        new = np.concatenate((x, y, np.abs(y), z), axis=-1)
        return new / 127., dset[1]

    output = doit(learning)
    if testing is not None:
        output += doit(testing)
    return output


def load_dataset(path=DEFAULT_DATASET, split=0., **kw):
    X, Y = pickle.load(gzip.open(path))
    dropoutliers = kw.get("drop_outliers", False)
    if kw.get("drop_outliers"):
        if not isinstance(dropoutliers, float) or dropoutliers <= 0. or dropoutliers >= 1.:
            dropoutliers = 0.95
        X, Y = drop_outliers(X, Y, dropoutliers)
    if kw.get("optimalish"):
        X, Y = optimalish_config((X, Y))
    if kw.get("as_matrix"):
        X = as_matrix(X)
    if kw.get("as_string"):
        Y = as_string(Y, kw.get("labels", labels))
    elif kw.get("as_onehot"):
        Y = as_onehot(Y, kw.get("categ", None))
    if split:
        return split_data(X, Y, split)
    return X, Y


# noinspection PyCallingNonCallable
def load_testsplit_dataset(boxer, **kw):
    lX, lY = load_dataset(f"{ltbroot}E_{boxer}_learning.pkl.gz", drop_outliers=kw.pop("drop_outliers", False), **kw)
    tX, tY = load_dataset(f"{ltbroot}E_{boxer}_testing.pkl.gz", **kw)
    return lX, lY, tX, tY


def normalize(X, mean=None, std=None, getparam=False):
    mean = X.mean(axis=0, keepdims=True) if mean is None else mean
    std = X.std(axis=0, keepdims=True) if std is None else std
    nX = (X - mean) / std
    return (nX, mean, std) if getparam else nX


def decorrelate(X, model=None, getmodel=False):
    from sklearn.decomposition import PCA
    model = PCA(whiten=True) if model is None else model
    lX = model.fit_transform(X)
    return (lX, model) if getmodel else lX

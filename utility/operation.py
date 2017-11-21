import gzip
import pickle

import numpy as np
from EBH.utility.const import labels, DEFAULT_DATASET
from EBH.utility.const import ltbroot


def average_filter(series, window=2):
    return np.convolve(series, np.ones(window) / window, mode="same")


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


def load_dataset(path=DEFAULT_DATASET, split=0., **kw):
    X, Y = pickle.load(gzip.open(path))
    dropJ = kw.get("dropJ")
    if dropJ:
        X, Y = drop_category(X, Y, 0, dropJ)
    if kw.get("as_matrix"):
        X = as_matrix(X)
    if kw.get("mxnormalize"):
        X /= 128.
    if kw.get("normalize"):
        X = normalize(X)
    if kw.get("as_matrix"):
        X = decorrelate(X)
    if kw.get("as_string"):
        Y = as_string(Y, kw.get("labels", labels))
    elif kw.get("as_onehot"):
        Y = as_onehot(Y, kw.get("categ", None))
    if split:
        return split_data(X, Y, split)
    return X, Y


# noinspection PyCallingNonCallable
def load_testsplit_dataset(boxer, **kw):
    lkw = dict(as_matrix=kw.get("as_matrix"), as_string=kw.get("as_string"),
               as_onehot=kw.get("as_onehot"), dropJ=kw.get("dropJ"))
    lX, lY = load_dataset(f"{ltbroot}E_{boxer}_learning.pkl.gz", **lkw)
    tX, tY = load_dataset(f"{ltbroot}E_{boxer}_testing.pkl.gz", **lkw)
    if kw.get("mxnormalize"):
        lX /= 128.
        tX /= 128.
    if kw.get("normalize"):
        lX, mu, sigma = normalize(lX, getparam=True)
        tX = normalize(tX, mu, sigma)
    if kw.get("decorrelate"):
        lX, pca = decorrelate(lX, getmodel=True)
        tX = decorrelate(tX, pca)
    ytr = kw.get("y_transform")
    if ytr is not None:
        lX[:, 1] = ytr(lX[:, 1])
        tX[:, 1] = ytr(tX[:, 1])
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

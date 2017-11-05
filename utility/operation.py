import gzip
import pickle

import numpy as np

from .const import projectroot, labels


def average_filter(series, window=2):
    return np.convolve(series, np.ones(window) / window, mode="same")


def load_dataset(flatten=True, strY=True, split=0.):
    X, Y = pickle.load(gzip.open(projectroot + "data.pkl.gz"))
    if flatten:
        X = X.reshape(len(X), np.prod(X.shape[1:], dtype=int))
    if strY:
        Y = np.vectorize(lambda ix: labels[ix])(Y)
    if split:
        arg = np.arange(len(X))
        np.random.shuffle(arg)
        n = int(len(X)*split)
        larg, targ = arg[n:], arg[:n]
        return X[larg], Y[larg], X[targ], Y[targ]
    return X, Y

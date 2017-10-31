import numpy as np


def average_filter(series, window=2):
    return np.convolve(series, np.ones(window) / window, mode="same")

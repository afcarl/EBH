import numpy as np


def find_peaks(data, threshold=75, center=True):
    peak = []
    peaks = []
    inpeak = False
    ndata = data.copy()
    ndata[ndata < threshold] = 0.
    for arg in np.arange(len(ndata)):
        d = ndata[arg]
        if not inpeak and not d:
            continue
        if ndata[arg]:
            inpeak = True
            peak.append(arg)
        else:
            if center:
                peaks.append(peak[0] + np.argmax(ndata[peak[0]:peak[-1]+1]))
            else:
                peaks.append(peak)
            peak = []
            inpeak = False
    return peaks


def extract_peak_proximity(peakarg, size, data, *more):
    hsize = size // 2
    Xs = [[d[p-hsize:d+hsize] for d in (data,) + more] for p in peakarg if len(data) - size > p >= hsize]
    return np.array(Xs)

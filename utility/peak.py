import numpy as np

from .operation import average_filter


def find_peaks(data, threshold=75, peaksize=None):
    peak = []
    peaks = []
    inpeak = False
    highpass = np.linalg.norm(data, axis=1) if data.ndim == 2 else data.copy()
    highpass[highpass < threshold] = 0.
    for arg, d in enumerate(highpass):
        if not inpeak and not d:
            continue
        if d:
            inpeak = True
            peak.append(arg)
        else:
            if peaks:
                if peak[0] - peaks[-1][-1] < 10:
                    last = peaks.pop()
                    peak = list(range(last[0], peak[-1]))
            peaks.append(peak)
            peak = []
            inpeak = False
    centers = np.array([p[0] + np.argmax(highpass[p[0]:p[-1]+1]) for p in peaks])
    if not peaksize or peaksize <= 1:
        return centers
    hsz = peaksize // 2
    return np.array([data[p-hsz:p+hsz] for p in centers if len(data)-hsz-1 > p > hsz])


def find_peaks_subtract(dw, threshtop=50, threshbot=None, filtersize=0, peaksize=None):
    threshbot = threshtop if threshbot is None else threshbot
    left = dw.get_data("left", norm=True)[-1] - dw.get_data("right", norm=True)[-1]
    if filtersize > 1:
        left = average_filter(left, window=filtersize)
    top = find_peaks(left, threshtop, peaksize)
    bot = find_peaks(-left, threshbot, peaksize)
    return top, bot


def extract_peaks(peakargs, data, proxy=5):
    prx = peakargs.copy()
    prx[:, 0] -= proxy
    prx[:, 1] += proxy
    out = [data[s:e] for s, e in prx]
    return out

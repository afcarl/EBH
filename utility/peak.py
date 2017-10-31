import numpy as np


def find_peaks(data, threshold=75, center=True):
    peak = []
    peaks = []
    inpeak = False
    highpass = np.linalg.norm(data, axis=1) if data.ndim == 2 else data.copy()
    highpass[highpass < threshold] = 0.
    for arg in range(len(highpass)):
        d = highpass[arg]
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

    if not center:
        return np.array([[p[0], p[-1]] for p in peaks])
    return np.array([p[0] + np.argmax(highpass[p[0]:p[-1]+1]) for p in peaks])


def find_peaks_subtract(dw, threshold=50, center=True):
    left = dw.get_data("left", norm=True)[-1] - dw.get_data("right", norm=True)[-1]
    top = find_peaks(left, threshold, center)
    bot = find_peaks(-left, threshold, center)
    return top, bot


def extract_peaks(peakargs, data, proxy=5):
    prx = peakargs.copy()
    prx[:, 0] -= proxy
    prx[:, 1] += proxy
    out = [data[s:e] for s, e in prx]
    return out

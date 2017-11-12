import numpy as np

from .operation import average_filter


def find_peaks(data, threshold=75, peaksize=10, mindist=10):
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
                if abs(peak[0] - peaks[-1][-1]) < mindist:
                    last = peaks.pop()
                    peak = list(range(last[0], peak[-1]))
            peaks.append(peak)
            peak = []
            inpeak = False

    hsz = peaksize // 2
    centers = np.array(
        [c for c in (p[0] + np.argmax(highpass[p[0]:p[-1]+1]) for p in peaks)
         if hsz < c < len(highpass)-hsz-1]
    )
    return centers


def find_peaks_subtract(dw, threshtop=50, threshbot=None, filtersize=0, peaksize=10, mindist=10):
    threshbot = threshtop if threshbot is None else threshbot
    left = dw.get_data("left", norm=True)[-1] - dw.get_data("right", norm=True)[-1]
    if filtersize > 1:
        left = average_filter(left, window=filtersize)
    top = find_peaks(data=left, threshold=threshtop, peaksize=peaksize, mindist=mindist)
    bot = find_peaks(data=-left, threshold=threshbot, peaksize=peaksize, mindist=mindist)
    return top, bot

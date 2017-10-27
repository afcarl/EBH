import numpy as np
from matplotlib import pyplot as plt

from parser import load_data


def plot_acceleration(ltime, ldata, rtime, rdata):
    fig, axarr = plt.subplots(2, 2)
    for i, (lcol, rcol, ax) in enumerate(zip(ldata.T, rdata.T, axarr.flat[:3]), start=1):
        ax.set_title(f"Axis {i}")
        ax.plot(ltime, lcol)
        ax.plot(rtime, rcol)
    axarr[-1, -1].plot(ltime, np.linalg.norm(ldata, axis=1), label="Left")
    axarr[-1, -1].plot(rtime, np.linalg.norm(rdata, axis=1), label="Right")
    axarr[-1, -1].set_title("Vector size (L2 norm)")
    plt.suptitle("Time vs. signal intensity")
    plt.figlegend()
    plt.show()


def plot_peak(time, data):
    fig, axarr = plt.subplots(2, 2)
    for i, (col, ax) in enumerate(zip(data.T, axarr.flat[:3]), start=1):
        ax.set_title(f"Axis {i}")
        ax.plot(time, col)
    axarr[-1, -1].plot(time, np.linalg.norm(data, axis=1))
    axarr[-1, -1].set_title("Vector size (L2 norm)")
    plt.suptitle("Time vs. signal intensity")
    plt.show()


def find_peaks(arr, ythresh, window=200, mindist=0):
    peaks = [0]
    blacklist = []
    ds = [0]
    for start in range(0, len(arr)-window):
        slc = arr[start:start+window]
        mx = slc.max()
        d = mx - slc.min()
        if d > ythresh:
            hits = start + np.argwhere(slc == mx).ravel()
            for hit in hits:
                if hit not in peaks + blacklist:
                    if hit - peaks[-1] < mindist:
                        if d > ds[-1]:
                            peaks[-1] = hit
                            ds[-1] = d
                        else:
                            blacklist.append(hit)
                        continue
                    peaks.append(hit)
                    ds.append(d)
            # peaks.add(start + np.where(slc == mx))
    return np.array(peaks[1:]).astype(int)


def plot_peaks(data):
    nY = np.linalg.norm(data, axis=1)
    peaks = find_peaks(nY, ythresh=9, window=9, mindist=6)
    plt.plot(data, nY)
    plt.plot(data[peaks], nY[peaks], "rx")
    plt.show()


def extract_peaks(data, size=10):
    nY = np.linalg.norm(data, axis=1)
    peaks = find_peaks(nY, ythresh=9, window=9, mindist=6)
    print("Found", len(peaks), "peaks")
    X = []
    hsize = size // 2
    for arg in filter(lambda p: len(nY)-hsize > p >= hsize, peaks):
        X.append(data[arg-hsize:arg+hsize])
    return np.array(X), peaks


if __name__ == '__main__':
    sz = 20
    hsz = sz//2
    lX, lY, rX, rY = load_data("sample")
    peak, peakarg = extract_peaks(lY, size=sz)
    for p, arg in zip(peak, peakarg):
        plot_peak(lX[arg-hsz:arg+hsz], p)

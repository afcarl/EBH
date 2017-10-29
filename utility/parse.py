import numpy as np

from EBH import projectroot


def _extract_line(line):
    # noinspection PyUnresolvedReferences
    line = line.strip()
    t = np.datetime64("T".join(line.split(" ")[:2]))
    a = np.array(line.split(" GRDATA ")[-1].split(" ")[-3:], dtype=int)
    return t, a


def _arrayify(stream, epoch0):
    times, accels = [], []
    for time, accel in map(_extract_line, stream):
        times.append(time - epoch0)
        accels.append(accel)
    return np.array(times), np.array(accels)


def extract_data(filepath, dumppath=None):

    lines = list(filter(lambda line: "GRDATA" in line, open(filepath)))
    left = filter(lambda line: "GRDATA L" in line, lines)
    right = filter(lambda line: "GRDATA R" in line, lines)

    lstart, _ = _extract_line(next(left))
    rstart, _ = _extract_line(next(right))

    (ltime, laccel), (rtime, raccel) = _arrayify(left, lstart), _arrayify(right, rstart)
    ltime, rtime = ltime.astype(float), rtime.astype(float)

    if dumppath:
        ltime.dump(dumppath + "sample_ltime.npa")
        rtime.dump(dumppath + "sample_rtime.npa")
        laccel.dump(dumppath + "sample_laccel.npa")
        raccel.dump(dumppath + "sample_raccel.npa")

    return ltime, laccel, rtime, raccel


if __name__ == '__main__':
    extract_data(projectroot + "logz/sample.txt", dumppath=projectroot + "npaz/")

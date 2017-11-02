import numpy as np

from EBH import projectroot
from utility.label import labels


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


def extract_data(filepath):
    lines = list(filter(lambda line: "GRDATA" in line or "TRAINER cmd: started" in line, open(filepath)))
    left = filter(lambda line: "GRDATA L" in line, lines)
    right = filter(lambda line: "GRDATA R" in line, lines)
    startline = [line for line in lines if "TRAINER cmd: started;" in line][0]
    epoch0 = np.datetime64("T".join(startline.split(" ")[:2]))

    (ltime, laccel), (rtime, raccel) = _arrayify(left, epoch0), _arrayify(right, epoch0)
    ltime, rtime = ltime.astype(float), rtime.astype(float)
    return ltime, laccel, rtime, raccel


def pull_annotation(filepath):
    with open(filepath) as handle:
        config, left, right = handle.read().replace(" ", "").split("\n")[:3]
    cfg = [tuple(c.split("-")) for c in config.split(":")[-1].split(";")]
    cfg = {k.strip(): int(v) for k, v in cfg}
    left = np.array([labels.index(l) for l in left.split(":")[-1] if l != "?"])
    right = np.array([labels.index(l) for l in right.split(":")[-1] if l != "?"])
    return left, right, cfg

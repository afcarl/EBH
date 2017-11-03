import numpy as np

from .const import labels


# noinspection PyUnresolvedReferences
def _extract_datetime(line):
    return np.datetime64("T".join(line.strip().split(" ")[:2])).astype(float)


def _extract_accel(line):
    return np.array(line.strip().split(" GRDATA ")[-1].split(" ")[-3:], dtype="int8")


def _extract_line(line):
    return _extract_datetime(line), _extract_accel(line)


def _arrayify(stream, start, end):
    times, accels = [], []
    for time, accel in map(_extract_line, stream):
        if time < start:
            continue
        times.append(time - start)
        accels.append(accel)
        if time >= end:
            break
    return np.array(times), np.array(accels)


def _extract_boundaries(lines, left, right):
    startline = [line for line in lines if "TRAINER cmd: started;" in line]
    if startline:
        epoch_start = _extract_datetime(startline[0])
    else:
        print("No startline! Falling back to first GRLINE...")
        epoch_start = min(_extract_datetime(left[0]), _extract_datetime(right[0]))
    endline = [line for line in lines if "remaining = 0\n" == line[-14:]]
    if endline:
        epoch_end = _extract_datetime(endline[0])
    else:
        print("No endline! Falling back to last GRLINE...")
        epoch_end = max(_extract_datetime(left[-1]), _extract_datetime(right[-1]))
    return epoch_start, epoch_end


def extract_data(filepath):
    lines = list(filter(lambda line: "GRDATA" in line or "TRAINER cmd:" in line, open(filepath)))
    left = list(filter(lambda line: "GRDATA L" in line, lines))
    right = list(filter(lambda line: "GRDATA R" in line, lines))

    epoch_start, epoch_end = _extract_boundaries(lines, left, right)

    ltime, laccel = _arrayify(left, epoch_start, epoch_end)
    rtime, raccel = _arrayify(right, epoch_start, epoch_end)
    return ltime, laccel, rtime, raccel


def pull_annotation(filepath):
    with open(filepath) as handle:
        config, left, right = handle.read().replace(" ", "").split("\n")[:3]
    cfg = [tuple(c.split("-")) for c in config.split(":")[-1].split(";")]
    cfg = {k.strip(): int(v) for k, v in cfg}
    left = np.array([labels.index(l) for l in left.split(":")[-1] if l != "?"])
    right = np.array([labels.index(l) for l in right.split(":")[-1] if l != "?"])
    return left, right, cfg

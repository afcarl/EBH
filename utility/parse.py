import numpy as np

from .const import labels


# noinspection PyUnresolvedReferences
def _extract_datetime(half):
    return np.datetime64("T".join(half.strip().split(" ")[:2])).astype(float)


def _extract_accel(half):
    hexa = bytearray.fromhex(half.strip().split("\t")[1].replace(" ", ""))
    assert len(hexa) == 20, "Invalid data: " + " ".join(hexa)
    split = hexa[5:10], hexa[10:15], hexa[15:]
    return np.stack([np.frombuffer(buf, dtype="int8") for buf in split])


def _arrayify(stream, start, end, clip=True):
    times, accels = np.array([]), np.array([np.zeros(3)])
    for line in stream:
        half1, half2 = line.strip().split(" rawdata: ")
        time = _extract_datetime(half1)
        if clip and time < start:
            continue
        acc = _extract_accel(half2)
        t = time - start
        times = np.concatenate((times, np.linspace(t, t-50, 5)))
        accels = np.concatenate((accels, np.stack(acc, axis=1)))
        if clip and time >= end:
            break
    return times[1:], accels[1:]


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


def extract_data(filepath, clip=True):
    lines = list(filter(lambda line: "rawdata:" in line or "TRAINER cmd:" in line, open(filepath)))
    left = list(filter(lambda line: "rawdata: LEFT" in line, lines))
    right = list(filter(lambda line: "rawdata: RIGHT" in line, lines))

    epoch_start, epoch_end = _extract_boundaries(lines, left, right)

    ltime, laccel = _arrayify(left, epoch_start, epoch_end, clip)
    rtime, raccel = _arrayify(right, epoch_start, epoch_end, clip)
    return ltime, laccel, rtime, raccel


def pull_annotation(filepath):
    with open(filepath) as handle:
        config, left, right = handle.read().replace(" ", "").split("\n")[:3]
    cfg = [tuple(c.split("-")) for c in config.split(":")[-1].split(";")]
    cfg = {k.strip(): int(v) for k, v in cfg}
    left = np.array([labels.index(l) for l in left.split(":")[-1] if l != "?"])
    right = np.array([labels.index(l) for l in right.split(":")[-1] if l != "?"])
    return left, right, cfg

import numpy as np

from .const import labels


# noinspection PyUnresolvedReferences
def _extract_datetime(half):
    return np.datetime64("T".join(half.strip().split(" ")[:2])).astype(float)


def _extract_accel(half, getseq=False):
    hexa = bytearray.fromhex(half.strip().split("\t")[1].replace(" ", ""))
    split = hexa[5:10], hexa[10:15], hexa[15:]
    accel = np.stack([np.frombuffer(buf, dtype="int8") for buf in split], axis=1)
    if not getseq:
        return accel
    return accel, int(np.frombuffer(hexa[1]))


def _arrayify_raw(stream, start, end, clip=True):
    accels, seqnum = [], []
    for line in stream:
        half1, half2 = line.strip().split(" rawdata: ")
        time = _extract_datetime(half1)
        if clip and time < start:
            continue
        acc, sn = _extract_accel(half2, getseq=True)
        seqnum.append(sn)
        accels.append(acc)
        if clip and time >= end:
            break
    assert seqnum[0] == 0
    assert len(np.unique(np.diff(seqnum))) == 1
    accels = np.array(accels)
    return np.concatenate(accels[np.argsort(seqnum)])


def _arrayify_accumulate(stream, start, end, clip=True):
    times, accels = [], []
    for line in stream:
        half1, half2 = line.strip().split(" rawdata: ")
        time = _extract_datetime(half1)
        if clip and time < start:
            continue
        acc = np.mean(_extract_accel(half2), axis=1).astype("int8")
        accels.append(acc)
        t = time - start
        times.append(t)
        if clip and time >= end:
            break
    assert len(times) == len(accels)
    return np.concatenate(times), np.concatenate(accels)


def _find_first_data_line(stream):
    for line in stream:
        if "rawdata:" in line:
            return [line]
    else:
        raise RuntimeError


def _extract_boundaries(lines):
    startline = [line for line in lines if "TRAINER cmd: started;" in line]
    endline = [line for line in lines if "remaining = 0\n" == line[-14:]]
    if not startline:
        print("No startline! Falling back to first rawdata...")
        startline = _find_first_data_line(lines)
    if not endline:
        print("No endline! Falling back to last rawdata...")
        endline = _find_first_data_line(lines[::-1])
    return _extract_datetime(startline[0]), _extract_datetime(endline[0])


def _isleft(logline):
    return "rawdata: LEFT" in logline


def extract_data(filepath, clip=True):

    lines = list(filter(lambda l: "rawdata:" in l or "TRAINER cmd:" in l, open(filepath)))
    epoch_start, epoch_end = _extract_boundaries(lines)

    # Data dim: [side (0, 1); N; rframe (0...4); axes (X, Y, Z)]
    times = []
    data = {True: [np.zeros((5, 3))], False: [np.zeros((5, 3))]}
    for line in filter(lambda l: "rawdata:" in l, lines):
        half1, half2 = line.strip().split(" rawdata: ")
        time = _extract_datetime(half1)
        if clip and time < epoch_start:
            continue
        acc = _extract_accel(half2)  # [rframe (0...4); axes (X, Y, Z)]
        side = _isleft(line)
        data[side].append(acc)
        data[not side].append(data[not side][-1])
        times.append(float(time - epoch_start))
        if clip and time > epoch_end:
            break
    data = np.stack((np.stack(data[True][1:]), np.stack(data[False][1:])))
    return np.array(times), data


def _rmstr(string, *sstrings):
    output = string[:]
    for ss in sstrings:
        output = output.replace(ss, "")
    return output


def pull_annotation(filepath):
    with open(filepath) as handle:
        config, left, right = _rmstr(handle.read(), " ", "?").split("\n")[:3]
    cfg = [tuple(c.split("-")) for c in config.split(":")[-1].split(";")]
    cfg = {k.strip(): int(v) for k, v in cfg}
    left = np.array([labels.index(l) for l in left.split(":")[-1]])
    right = np.array([labels.index(l) for l in right.split(":")[-1]])
    return left, right, cfg

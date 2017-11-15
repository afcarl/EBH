from EBH.utility.operation import load_dataset
from EBH.utility.frame import DataWrapper


def inspect_session(ID):
    dw = DataWrapper(ID)
    lX, rX = dw.get_peaks(peaksize=10)
    lY, rY = dw.get_annotations("left"), dw.get_annotations("right")

    print("Data shapes in", ID)
    print("lX:", lX.shape)
    print("lY:", lY.shape)
    print("rX:", rX.shape)
    print("rY:", rY.shape)


def inspect_classes():
    from csxdata.stats import normaltest, inspection
    from csxdata.visual.histogram import fullplot
    names = []
    for l in "XYZ":
        for i in range(10):
            names.append(l + str(i))

    X, Y = load_dataset(flatten=True)
    inspection.category_frequencies(Y)
    inspection.correlation(X, names=names)
    normaltest.full(X, names=names)
    for name, column in zip(names, X.T):
        fullplot(column, name)


def compare_parsers(onfile):
    from EBH.utility.parse_zolaly import extract_data as zed
    from EBH.utility.parse import extract_data as ced
    from EBH.utility.const import logroot
    zltime, zldata, zrtime, zrdata = zed(logroot + onfile)
    cltime, cldata, crtime, crdata = ced(logroot + onfile)
    print("COMARING ON", onfile)
    print(f"LTIMES (Z/C): {zltime.size} / {cltime.size}")
    print(f"RTIMES (Z/C): {zrtime.size} / {crtime.size}")
    print(f"LSHAPE (Z/C): {zldata.shape} / {cldata.shape}")
    print(f"RSHAPE (Z/C): {zrdata.shape} / {crdata.shape}")
    d = zldata - cldata
    print("SUMD:", d.sum())
    print()

    assert zltime.size == cltime.size
    assert zrtime.size == crtime.size
    assert zldata.shape == cldata.shape
    assert zrdata.shape == crdata.shape
    # assert not d.sum()


def read_counters(onfile):
    import numpy as np
    from matplotlib import pyplot as plt
    from EBH.utility.const import logroot
    lines = [line for line in open(logroot + onfile).read().split("\n") if "rawdata:" in line]
    lcount = np.frombuffer(bytearray.fromhex(
        "".join([line.split("\t")[1].split(" ")[1] for line in lines if "rawdata: LEFT" in line])), "uint8")
    rcount = np.frombuffer(bytearray.fromhex(
        "".join([line.split("\t")[1].split(" ")[1] for line in lines if "rawdata: RIGHT" in line])), "uint8")
    N = min(len(lcount), len(rcount))
    plt.plot(np.abs(lcount[:N] - rcount[:N]), "r.")
    plt.show()


if __name__ == '__main__':
    inspect_session("Virginia_le")

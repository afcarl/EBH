import numpy as np
from EBH.utility.operation import load_dataset, decorrelate, as_matrix
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
    for l in "YP":
        for i in range(10):
            names.append(l + str(i))

    X, Y = load_dataset(as_matrix=False, as_string=True)
    X = decorrelate(X.transpose(0, 2, 1).reshape(X.shape[0], X.shape[1] * X.shape[2]))

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


def generate_neighborhood():
    from EBH.utility.assemble import assemble_data
    from EBH.utility.const import projectroot
    X, Y = assemble_data(peaksize=20)
    x, y, z = X.T
    X = np.concatenate((x.T, y.T, np.abs(y.T), z.T), axis=-1)
    X = as_matrix(X)
    Xs, Ys = X.astype("int8").tostring(), Y.astype("int8").tostring()
    Ns = np.array([len(X)], dtype="int16").tostring()
    with open(projectroot + "neighborhood.bin", "wb") as handle:
        handle.write(Ns + Ys + Xs)


if __name__ == '__main__':
    inspect_session("box4_fel")

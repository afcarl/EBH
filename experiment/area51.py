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


if __name__ == '__main__':
    inspect_classes()

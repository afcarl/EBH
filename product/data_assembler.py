import numpy as np
import pandas as pd

from EBH.utility.assemble import dwstream
from EBH.utility.const import professionals
from EBH.utility.operation import as_string
from EBH.utility.axisconfig import DataConfig


LABELS = ["name", "orientation", "ID", "professional", "hand"]


def assemble(config: DataConfig):
    Ydf = pd.DataFrame(columns=["gesture", "boxer", "orient", "ID", "pro", "hand"])
    Xdf = pd.DataFrame(columns=[f"P{i:0>2}" for i in range(config.PEAKSIZE)])
    for dw in dwstream():
        name, orient, ID, pro, hand = [], [], [], [], []
        tx, ty, bx, by = dw.get_learning_table(config.PEAK, readingframe=0, splitsides=True)
        boundary = len(tx)
        x, y = np.r_[tx, bx], np.r_[ty, by]
        mask = y < 3
        x, y = x[mask], y[mask]
        N = len(x)
        name.append([dw.boxer]*N)
        orient.append([dw.orientation]*N)
        ID.append([dw.ID]*N)
        pro.append([dw.boxer in professionals]*N)
        hand.append(["left"]*boundary + ["right"]*(N - boundary))
        idx = pd.RangeIndex(0, N)
        Ydf.append(pd.DataFrame(data=dict(gesture=as_string(y), boxer=name, orient=orient, ID=ID, pro=pro, hand=hand),
                                columns=Ydf.columns, index=idx))
        Xdf.append(pd.DataFrame(data=np.array(x), columns=Xdf.columns, index=idx))
    config.N = len(Xdf)
    return pd.concat((Ydf, Xdf))


if __name__ == '__main__':

    massive_df = assemble(DataConfig(-1, ))

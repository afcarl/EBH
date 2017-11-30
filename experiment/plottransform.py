import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FastICA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE, SpectralEmbedding, Isomap

from csxdata.visual.scatter import Scatter3D, Scatter2D

from EBH.utility.assemble import dwstream
from EBH.utility.operation import as_matrix, as_string, optimalish_config
from EBH.utility.const import professionals, projectroot


def combine_categories(Y1, Y2):
    return np.vectorize(lambda s1, s2: "_".join(map(str, (s1, s2))))(Y1, Y2)


def load_data(usecache=True):
    cachefile = f"{projectroot}plotcache.cch"
    if usecache and os.path.exists(cachefile):
        return pickle.load(open(cachefile, "rb"))
    Xs, Ys, name, hand, ID, pro = [], [], [], [], [], []
    for dw in dwstream():
        x, y = dw.get_learning_table(PEAKSIZE, READINGFRAME)
        mask = y < 3
        x, y = x[mask], y[mask]
        Xs.append(x)
        Ys.append(y)
        N = len(x)
        name.append([dw.boxer]*N)
        hand.append([dw.orientation]*N)
        ID.append([dw.ID]*N)
        pro.append([dw.boxer in professionals]*N)
    Xs, Ys, name, hand, ID, pro = map(np.concatenate, (Xs, Ys, name, hand, ID, pro))
    Ys = as_string(Ys)
    Xs, Ys = optimalish_config((Xs, Ys))
    output = as_matrix(Xs), Ys, name, hand, ID, pro
    if usecache:
        pickle.dump(output, open(cachefile, "wb"))
    return output


def get_model(model, ndim):
    m = {"lda": LDA(n_components=ndim), "pca": PCA(n_components=ndim, whiten=True),
         "kpca": KernelPCA(n_components=ndim, kernel="rbf"),
         "ica": FastICA(n_components=ndim), "t-sne": TSNE(ndim),
         "se": SpectralEmbedding(ndim, n_jobs=4),
         "isomap": Isomap(n_components=ndim, n_jobs=4,)}[model]
    return m


def plot_transformation3D(transform, X, Y=None):
    model = get_model(transform, 3)
    latent = model.fit_transform(X, Y)
    lX = latent[0] if len(latent) == 2 else latent
    scat = Scatter3D(lX, Y, ["LF01", "LF02", "LF03"])
    scat.split_scatter(show=True, legend=True)


def plot_transformation2D(transform, X, Y=None):
    model = get_model(transform, 2)
    latent = model.fit_transform(X, Y)
    lX = latent[0] if len(latent) == 2 else latent
    scat = Scatter2D(lX, Y, axlabels=["LF01", "LF02"])
    scat.split_scatter(show=False, center=False, label=False, alpha=0.3)
    plt.legend()
    plt.show()


PEAKSIZE = 20
READINGFRAME = 0


def main():
    Xs, gesture, name, hand, ID, pro = load_data(usecache=True)
    plot_transformation2D("lda", Xs, gesture)


if __name__ == '__main__':
    main()

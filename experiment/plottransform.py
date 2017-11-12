import gzip
import pickle

import numpy as np
from sklearn.decomposition import PCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from csxdata.visual.scatter import Scatter3D

from EBH.utility.operation import load_dataset


def get_model(model, ndim):
    m = {"lda": LDA(n_components=ndim), "pca": PCA(n_components=ndim, whiten=True),
         "ica": FastICA(n_components=ndim)}[model]
    return m


def plot_transformation(transform, X, Y=None):
    model = get_model(transform, 3)
    lX = model.fit_transform(X, Y)
    arg = np.arange(len(lX))
    np.random.shuffle(arg)
    arg = arg[:len(arg)//10]
    scat = Scatter3D(lX[arg], Y[arg], ["LF01", "LF02", "LF03"])
    scat.split_scatter(show=True, legend=True)


if __name__ == '__main__':
    plot_transformation("ica", *load_dataset(as_matrix=True, normalize=True, as_string=True))

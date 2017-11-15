from sklearn.decomposition import PCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE, SpectralEmbedding, Isomap

from csxdata.visual.scatter import Scatter3D, Scatter2D

from EBH.utility.operation import load_dataset


def get_model(model, ndim):
    m = {"lda": LDA(n_components=ndim), "pca": PCA(n_components=ndim, whiten=True),
         "ica": FastICA(n_components=ndim), "t-sne": TSNE(ndim),
         "se": SpectralEmbedding(ndim, n_jobs=4),
         "isomap": Isomap(n_components=ndim, n_jobs=4)}[model]
    return m


def plot_transformation3D(transform, X, Y=None):
    model = get_model(transform, 3)
    lX = model.fit_transform(X, Y)
    scat = Scatter3D(lX, Y, ["LF01", "LF02", "LF03"])
    scat.split_scatter(show=True, legend=True)


def plot_transformation2D(transform, X, Y=None):
    model = get_model(transform, 2)
    lX = model.fit_transform(X, Y)
    scat = Scatter2D(lX, Y, axlabels=["LF01", "LF02"])
    scat.split_scatter(show=True, center=True, label=True)


if __name__ == '__main__':
    plot_transformation2D("t-sne", *load_dataset(as_matrix=True, normalize=True, as_string=True))

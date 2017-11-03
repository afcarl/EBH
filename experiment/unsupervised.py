import os

import numpy as np
from sklearn.decomposition import PCA, FastICA, KernelPCA
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from keras.models import Model
from keras.layers import Dense, Input

from EBH.utility import frame
from EBH.utility.const import logroot, projectroot


def get_full_data(usecache=True, appendnorm=True):
    if os.path.exists(projectroot + "cch/fullcache.npa") and usecache:
        return np.load(projectroot + "cch/fullcache.npa")
    data = []
    for file in sorted(os.listdir(logroot)):
        dw = frame.DataWrapper(logroot + file)
        print(dw.ID)
        data.append(dw.get_peaks_vanilla(appendnorm=appendnorm))
    data = np.concatenate(data, axis=0)
    if usecache:
        data.dump(projectroot + "cch/fullcache.npa")
    return data


def get_subtracted_data(usecache=True, appendnorm=False):
    if os.path.exists(projectroot + "cch/subtrcache.npa") and usecache:
        return np.load(projectroot + "cch/subtrcache.npa")
    data = []
    for file in sorted(os.listdir(logroot)):
        dw = frame.DataWrapper(logroot + file)
        print(dw.ID)
        toppeaks, botpeaks = dw.get_peaks(appendnorm)
        print("RIGHT:", toppeaks.shape)
        print("LEFT: ", botpeaks.shape)
        data.append(toppeaks)
        data.append(botpeaks)
    data = np.concatenate(data, axis=0)
    if usecache:
        data.dump(projectroot + "cch/subtrcache.npa")
    return data


def plot_transform(data, trname):
    model = {"pca": PCA(whiten=True), "ica": FastICA(whiten=True), "kpca": KernelPCA(kernel="rbf")}[trname.lower()]
    lX = model.fit_transform(data / 255.).T
    if trname in "kpca":
        print("EXPLAINED VARIANCE:", model.explained_variance_ratio_)
    plt.plot(lX[0], lX[1], "b.", alpha=0.25)
    plt.title(trname.upper())
    plt.show()


def plot_decboundaries(lX, model):
    x_min, x_max = lX[:, 0].min() - 1, lX[:, 0].max() + 1
    y_min, y_max = lX[:, 1].min() - 1, lX[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, .02), np.arange(y_min, y_max, .02))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    if isinstance(model, Model):
        Z = Z.argmax(axis=-1)
    Z = Z.reshape(xx.shape)
    plt.imshow(Z, interpolation='nearest', aspect='auto', origin='lower',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()))
    plt.plot(lX[:, 0], lX[:, 1], 'k.', markersize=2)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()


def fit_kmeans(data):
    lX = PCA(whiten=True, n_components=2).fit_transform(data / 255.)
    model = KMeans(n_clusters=3).fit(lX)
    plot_decboundaries(lX, model=model)


def get_autoencoder(data):
    inputs = Input(data.shape[1:])
    encoder = Dense(120, activation="relu")(inputs)
    encoder = Dense(30, activation="relu")(encoder)
    enc_out = Dense(4, activation="relu")(encoder)
    encmodel = Model(inputs, enc_out, name="Encoder")

    decoder = Dense(4, activation="relu")(encoder)
    decoder = Dense(30, activation="relu")(decoder)
    decoder = Dense(120, activation="relu")(decoder)
    dec_out = Dense(data.shape[1])(decoder)

    autoencoder = Model(inputs, outputs=dec_out)
    autoencoder.compile(optimizer="adam", loss="mse")
    return encmodel, autoencoder


def fit_autoencoder(data):
    lX = PCA(2, whiten=True).fit_transform(data)
    enc, ae = get_autoencoder(lX)
    ae.fit(lX, lX, batch_size=50, epochs=120, validation_split=0.1)
    plot_decboundaries(lX, model=enc)


def fit_gmm(data):
    from sklearn.mixture import GaussianMixture as MModel
    lX = PCA(n_components=2, whiten=True).fit_transform(data)
    model = MModel(3, covariance_type="full").fit(lX)
    plot_decboundaries(lX, model)


def fit_affinityprop(data):
    from sklearn.cluster import AffinityPropagation
    lX = PCA(n_components=2, whiten=True).fit_transform(data)
    model = AffinityPropagation().fit(lX)
    plot_decboundaries(lX, model)


if __name__ == '__main__':
    X = get_full_data(usecache=False, appendnorm=False)
    X = X.reshape(X.shape[0], np.prod(X.shape[1:]))
    print("FINAL X SIZE:", X.shape)

    # fit_autoencoder(X)
    # fit_gmm(X)
    # fit_kmeans(X)
    fit_affinityprop(X)
    # plot_transform(X, "pca")

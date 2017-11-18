import numpy as np
from sklearn.utils import compute_class_weight

from keras.models import Sequential, Model
from keras.layers import BatchNormalization, Dense, Input, Concatenate
from keras.layers import Conv1D, Flatten, PReLU, Activation, MaxPooling1D
from keras.layers import LSTM as RLayer
from keras.optimizers import Adam

from EBH.utility.operation import load_testsplit_dataset
from EBH.utility.visual import plot_learning_dynamics


def get_shaped_data(boxer="Virginia"):
    lX, lY, tX, tY = load_testsplit_dataset(boxer, as_onehot=True, normalize=True)
    print(f"X shape: {lX.shape[1:]}")
    print(f"Y shape: {lY.shape[1:]}")
    print(f"N learning:", len(lX))
    print(f"N testing ({boxer}):", len(tX))
    return lX, lY, tX, tY


def get_simple_convnet(inshape, outshape):
    ann = Sequential(layers=[
        BatchNormalization(input_shape=inshape),
        Conv1D(24, kernel_size=5, kernel_regularizer="l2"), Activation("relu"),  # 6
        Conv1D(12, kernel_size=3, kernel_regularizer="l2"),  # 4
        MaxPooling1D(), Flatten(), Activation("relu"),  # 2 x 8 = 16
        Dense(30, activation="relu", kernel_regularizer="l2"), BatchNormalization(),
        Dense(outshape[0], activation="softmax", kernel_regularizer="l2")
    ])
    ann.compile(optimizer="nadam", loss="categorical_crossentropy", metrics=["acc"])
    return ann


def get_fully_convnet(inshape, outshape):
    ann = Sequential(layers=[
        BatchNormalization(input_shape=inshape),
        Conv1D(24, kernel_size=3, kernel_regularizer="l2"), MaxPooling1D(), Activation("relu"),  # 4
        Conv1D(3, kernel_size=5, kernel_regularizer="l2"), Flatten(), Activation("softmax")
    ])


def get_hydra_convnet(inshape, outshape):
    x = Input(inshape)  # 10, 2

    bn = BatchNormalization()(x)

    c1 = PReLU()(Conv1D(64, kernel_size=3)(bn))  # 8, 64
    c1 = BatchNormalization()(c1)
    c1 = PReLU()(Conv1D(32, kernel_size=3)(c1))  # 6, 32
    c1 = BatchNormalization()(c1)
    c1 = PReLU()(Conv1D(32, kernel_size=3)(c1))  # 4, 32
    c1 = BatchNormalization()(c1)
    c1 = Flatten()(Conv1D(16, kernel_size=3)(c1))  # 2x16 = 32

    c2 = PReLU()(Conv1D(32, kernel_size=5)(bn))  # 6, 32
    c2 = BatchNormalization()(c2)
    c2 = Flatten()(Conv1D(32, kernel_size=5)(c2))  # 2x32 = 64

    c3 = Flatten()(Conv1D(64, kernel_size=7)(bn))  # 2x64 = 128

    c = BatchNormalization()(PReLU()(Concatenate()([c1, c2, c3])))  # 224
    d = PReLU()(Dense(80)(c))  # 80
    d = BatchNormalization()(d)
    d = Dense(12, activation="relu")(d)  # 12
    d = BatchNormalization()(d)
    o = Dense(outshape[0], activation="softmax")(d)

    ann = Model(inputs=x, outputs=o, name="Hydra")
    ann.compile(optimizer=Adam(0.1), loss="categorical_crossentropy", metrics=["acc"])
    return ann


def get_lstm(inshape, outshape):
    ann = Sequential(layers=[
        BatchNormalization(input_shape=inshape),
        RLayer(120, activation="relu", return_sequences=False, kernel_regularizer="l2"),
        Dense(60, activation="tanh", kernel_regularizer="l2"),
        Dense(outshape[0], activation="softmax")
    ])
    ann.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["acc"])
    return ann


def xperiment():
    lX, lY, tX, tY = get_shaped_data("Virginia")
    cls = np.unique(lY.argmax(axis=1))
    w = compute_class_weight("balanced", cls, lY.argmax(axis=1))
    net = get_simple_convnet(lX.shape[1:], lY.shape[1:])
    history = net.fit(lX, lY, batch_size=32, epochs=300, verbose=True, validation_data=(tX, tY),
                      class_weight=w)
    plot_learning_dynamics(history)


if __name__ == '__main__':
    xperiment()

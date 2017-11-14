from keras.models import Sequential
from keras.layers import Dense

from EBH.utility.operation import load_dataset
from EBH.utility.visual import plot_learning_dynamics

lX, lY, tX, tY = load_dataset(split=0.1, as_matrix=True, as_onehot=True, normalize=True)

indim, outdim = lX.shape[-1], lY.shape[-1]

print(f"ANN fitting data with dimensionality: {indim} / {outdim}")

ann = Sequential(layers=[
    Dense(180, input_dim=indim, activation="relu", kernel_regularizer="l2"),
    Dense(40, activation="relu", kernel_regularizer="l2"),
    Dense(outdim, activation="softmax")
])
ann.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["acc"])

history = ann.fit(lX, lY, batch_size=32, epochs=200,
                  validation_data=(tX, tY),
                  class_weight="balanced", verbose=False)

plot_learning_dynamics(history)

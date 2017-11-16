from keras.models import Sequential
from keras.layers import Dense

from csxdata.utilities.vectorop import split_by_categories

from EBH.utility.const import labels, ltbroot
from EBH.utility.operation import load_dataset
from EBH.utility.visual import plot_learning_dynamics


def build_ann(indim, outdim):
    print(f"Building ANN for data with dimensionality: {indim} / {outdim}")
    ann = Sequential(layers=[
        Dense(60, input_dim=indim, activation="tanh"),
        Dense(10, activation="tanh"),
        Dense(outdim, activation="softmax")
    ])
    ann.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["acc"])
    return ann


def split_experiment(learn_on_me, test_on_me, plot=True):
    lX, lY = learn_on_me
    ann = build_ann(lX.shape[-1], lY.shape[-1])
    history = ann.fit(lX, lY, batch_size=32, epochs=60,
                      validation_data=test_on_me,
                      verbose=False, class_weight="balanced")

    reporttmpl = "Evaluation on {}: cost: {} acc: {}"

    bycat = split_by_categories(test_on_me[-1].argmax(axis=1))
    for cat, arg in bycat.items():
        cost, acc = ann.evaluate(test_on_me[0][arg], test_on_me[1][arg], verbose=False)
        print(reporttmpl.format(labels[cat], cost, acc))

    cost, acc = ann.evaluate(*test_on_me, verbose=False)
    print(reporttmpl.format("ALL", cost, acc))
    if plot:
        plot_learning_dynamics(history)


def basic_validation():
    lX, lY, vX, vY = load_dataset(split=0.1, as_matrix=True, as_onehot=True, normalize=True)
    split_experiment(learn_on_me=(lX, lY), test_on_me=(vX, vY))


def advanced_testing():
    loadkw = dict(as_matrix=True, as_onehot=True, normalize=True)
    split_experiment(
        learn_on_me=load_dataset(ltbroot + "learning.pkl.gz", **loadkw),
        test_on_me=load_dataset(ltbroot + "testing.pkl.gz", **loadkw)
    )


if __name__ == '__main__':
    advanced_testing()

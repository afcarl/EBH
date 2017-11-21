from keras.models import Sequential
from keras.layers import Dense

from csxdata.utilities.vectorop import split_by_categories

from EBH.utility.const import labels, ltbroot, boxer_names
from EBH.utility.operation import load_dataset, load_testsplit_dataset
from EBH.utility.visual import plot_learning_dynamics


def build_ann(indim, outdim):
    # print(f"Building ANN for data with dimensionality: {indim} / {outdim}")
    ann = Sequential(layers=[
        Dense(12, input_dim=indim, activation="tanh", kernel_regularizer="l2"),
        Dense(12, input_dim=indim, activation="tanh", kernel_regularizer="l2"),
        Dense(outdim, activation="softmax")
    ])
    ann.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["acc"])
    return ann


def split_experiment(learn_on_me, test_on_me, plot=True, verbose=1):
    lX, lY = learn_on_me
    tX, tY = test_on_me
    ann = build_ann(lX.shape[-1], lY.shape[-1])

    history = ann.fit(lX, lY, batch_size=16, epochs=100,
                      validation_data=test_on_me,
                      verbose=False, class_weight="balanced")

    reporttmpl = "Evaluation on {}: cost: {} acc: {}"

    bycat = split_by_categories(tY.argmax(axis=1))
    bycatacc = dict()
    for cat, arg in bycat.items():
        cost, acc = ann.evaluate(tX[arg], tY[arg], verbose=False)
        bycatacc[cat] = acc
        if verbose:
            print(reporttmpl.format(labels[cat], cost, acc))

    cost, acc = ann.evaluate(tX, tY, verbose=False)
    bycatacc["ALL"] = acc
    if verbose:
        print(reporttmpl.format("ALL", cost, acc))
    if plot:
        plot_learning_dynamics(history)
    return bycatacc


def basic_validation():
    lX, lY, vX, vY = load_dataset(split=0.1, as_matrix=True, as_onehot=True, mxnormalize=True)
    split_experiment(learn_on_me=(lX, lY), test_on_me=(vX, vY))


def basic_testing(boxer="Bela"):
    lX, lY, tX, tY = load_testsplit_dataset(boxer, mxnormalize=True, as_onehot=True, as_matrix=True)
    split_experiment(learn_on_me=(lX, lY), test_on_me=(tX, tY))


def advanced_testing():
    for name in boxer_names:
        lX, lY, tX, tY = load_testsplit_dataset(name, as_onehot=True, as_matrix=True)
        bycat_acc = split_experiment((lX, lY), (tX, tY), plot=False, verbose=0)
        print("-"*50)
        print("Split evaluation on excluded sample:", name)
        for cat in range(3):
            print(f"{labels[cat]}: {bycat_acc[cat]:.2%}")
        print(f"ALL: {bycat_acc['ALL']:.2%}")


if __name__ == '__main__':
    advanced_testing()

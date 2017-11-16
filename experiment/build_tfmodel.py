import numpy as np
import tensorflow as tf

from EBH.utility.operation import load_dataset, shuffle, split_data


class TFPerceptron:

    def __init__(self, indim, hiddens, outdim):
        self.X = tf.placeholder(tf.float32, [None, indim], name="inputs")
        self.Y = tf.placeholder(tf.float32, [None, outdim], name="labels")

        Z = self._build_computational_graph_middle(hiddens, outdim)

        self.output = tf.nn.softmax(Z, name="output")
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z, labels=self.Y, name="xent"))
        eq = tf.equal(tf.argmax(Z, 1), tf.argmax(self.Y, 1))
        self.acc = tf.reduce_mean(tf.cast(eq, tf.float32), name="acc")
        self._train_step = tf.train.AdamOptimizer().minimize(self.cost, name="trainstep")
        self.session = tf.InteractiveSession()
        tf.global_variables_initializer().run()

    def _build_computational_graph_middle(self, hiddens, outdim):
        dense = tf.layers.dense
        A = dense(self.X, hiddens[0], activation=tf.nn.tanh, kernel_regularizer=tf.nn.l2_loss)
        for h in hiddens[1:]:
            A = dense(A, h, activation=tf.nn.tanh, kernel_regularizer=tf.nn.l2_loss)
        return dense(A, outdim, activation=None, kernel_regularizer=tf.nn.l2_loss)

    def fit(self, X, Y, batch_size=32, epochs=60, validation=(), verbose=1):
        for epoch in range(1, epochs+1):
            if verbose:
                print("-"*50)
                print(f"Epoch {epoch}")
            X, Y = shuffle(X, Y)
            self._epoch(X, Y, batch_size, validation, verbose)

    def _epoch(self, X, Y, batch_size=32, validation=(), verbose=1):
        N = len(X)
        strln = len(str(N))
        stream = ((X[start:start+batch_size], Y[start:start+batch_size])
                  for start in range(0, len(X), batch_size))
        done = 0
        for x, y in stream:
            cost = self._fit_batch(x, y)
            done += len(x)
            if verbose:
                print(f"\rTraining... {done:>{strln}}/{N} Cost: {cost:.4f}", end="")
        if verbose:
            print()
        if validation and verbose:
            vcost, vacc = self.evaluate(*validation)
            print(f"Validation cost: {vcost:.4f}, acc: {vacc:.2%}")

    def _fit_batch(self, X, Y):
        return self.session.run([self.cost, self._train_step], feed_dict={self.X: X, self.Y: Y})[0]

    def predict(self, X):
        return self.session.run(self.output, feed_dict={self.X: X})

    def evaluate(self, X, Y):
        cost = self.session.run(self.cost, feed_dict={self.X: X, self.Y: Y}).mean()
        acc = self.session.run(self.acc, feed_dict={self.X: X, self.Y: Y})
        return cost, acc


# def leave_one_out_testing():
#     acc = []
#     for name in names:
#         lX, lY = load_dataset()



def main():
    X, Y = load_dataset(split=0., as_matrix=True, as_onehot=True, normalize=True)
    accs = []
    for run in range(100):
        print(f"RUN {run}", end=" ")
        lX, lY, tX, tY = split_data(X, Y, alpha=0.5, shuff=True)
        ann = TFPerceptron(lX.shape[-1], hiddens=[60, 10], outdim=lY.shape[-1])
        ann.fit(lX, lY, epochs=180, validation=(tX, tY), verbose=False)
        evalcost, evalacc = ann.evaluate(tX, tY)
        accs.append(evalacc)
        print(f"cost: {evalcost:.4f}, acc: {evalacc:.2%}")
    print("*"*50)
    print("FINAL EVALUATION")
    print("acc mean:", np.mean(accs))
    print("acc std:", np.std(accs))
    print("acc min:", np.min(accs))
    print("acc max:", np.max(accs))


if __name__ == '__main__':
    main()

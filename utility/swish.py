from keras.engine.topology import Layer
from keras import backend as K


class Swish(Layer):

    def build(self, input_shape):
        self.beta = self.add_weight("beta", shape=[1], initializer="ones")
        super().build(input_shape)

    def call(self, x):
        return x * K.sigmoid(self.beta * x)

    def compute_output_shape(self, input_shape):
        return input_shape

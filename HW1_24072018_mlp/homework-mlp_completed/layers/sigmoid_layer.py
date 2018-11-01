
import numpy as np


class SigmoidLayer:
    def __init__(self):
        """Applies the element-wise function :math:`f(x) = 1 / ( 1 + exp(-x))`
        """
        self.trainable = False

    def forward(self, Input):
        # TODO: Put your code here
        # Please delete `pass` and return the output
        self.output = 1 / (1 + np.exp(-Input))
        return self.output

    def backward(self, delta):
        # TODO: Put your code here
        # Please delete `pass`, calculate and return delta
        delta_l = self.output * (1 - self.output) * delta
        return delta_l

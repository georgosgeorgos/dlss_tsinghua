"""relu layer"""

import numpy as np


class ReLULayer:
    def __init__(self):
        """Applies the rectified linear unit function element-wise :math:`{ReLU}(x)= max(0, x)`
        """
        self.trainable = False

    def forward(self, Input):
        # TODO: Put your code here
        # Please delete `pass` and return the output
        output = Input.copy()
        output[output < 0] = 0
        self.cache = output.copy()
        return output

    def backward(self, delta):
        # TODO: Put your code here
        # Please delete `pass`, calculate and return delta
        delta[self.cache == 0] = 0
        return delta

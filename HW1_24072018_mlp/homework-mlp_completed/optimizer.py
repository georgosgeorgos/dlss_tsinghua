import numpy as np


class SGD(object):
    def __init__(self, learning_rate, weight_decay):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def step(self, model):
        layers = model.layer_list
        for layer in layers:
            if layer.trainable:
                # Calculate diff_W and diff_b
                layer.diff_W = -self.learning_rate * (layer.grad_W + self.weight_decay * layer.W)
                layer.diff_b = -self.learning_rate * layer.grad_b

                # weight updating
                layer.W += layer.diff_W
                layer.b += layer.diff_b


class SGDwithMomentum(object):
    def __init__(self, learning_rate, weight_decay, momentum):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.count = 0
        self.k = 100

    def step(self, model):
        layers = model.layer_list
        for layer in layers:
            if layer.trainable:
                # TODO: Calculate diff_W and diff_b with momentum
                if self.count > self.k:
                    layer.diff_W = (
                        -self.learning_rate * (layer.grad_W + self.weight_decay * layer.W)
                        + self.momentum * layer.diff_W
                    )
                    layer.diff_b = -self.learning_rate * layer.grad_b + self.momentum * layer.diff_b
                else:
                    layer.diff_W = (
                        -self.learning_rate * (layer.grad_W + self.weight_decay * layer.W) + 0.5 * layer.diff_W
                    )
                    layer.diff_b = -self.learning_rate * layer.grad_b + 0.5 * layer.diff_b
                self.count += 1

                # weight updating
                layer.W += layer.diff_W
                layer.b += layer.diff_b

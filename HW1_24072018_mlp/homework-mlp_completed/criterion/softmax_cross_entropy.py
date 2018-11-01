"""softmax cross entropy loss layer"""

import numpy as np


class SoftmaxCrossEntropy:
    def __init__(self):
        self.acc = 0
        self.loss = np.zeros(1, dtype='f')

    def forward(self, logit, gt):
        self.Input = gt
        # Softmax
        eps = 1e-9  # a small number to prevent dividing by zero
        exp_thetaX = np.exp(logit)
        self.p = exp_thetaX / (eps + exp_thetaX.sum(axis=1, keepdims=True))

        # calculate the accuracy
        predict_y = np.argmax(self.p, axis=1)
        gt_y = np.argmax(gt, axis=1)
        com = predict_y == gt_y
        self.acc = np.mean(com)

        # calculate the loss
        tmp = -np.log(np.diag(np.dot(gt, self.p.T)) + 10 ** -12)
        self.loss = np.mean(tmp)
        return self.loss

    def backward(self):
        # calculate delta
        self.delta = self.p - self.Input
        return self.delta

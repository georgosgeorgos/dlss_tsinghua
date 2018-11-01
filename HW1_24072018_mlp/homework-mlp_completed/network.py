

class Network:
    def __init__(self):
        self.layer_list = []
        self.num_layer = 0

    def add(self,layer):
        self.num_layer = self.num_layer + 1
        self.layer_list.append(layer)

    def forward(self, x):
        for i in range(self.num_layer):
            x = self.layer_list[i].forward(x)

        return x

    def backward(self, delta):
        for i in reversed(range(self.num_layer)):
            delta = self.layer_list[i].backward(delta)

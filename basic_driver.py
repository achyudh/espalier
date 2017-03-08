import numpy as np

import layers
import loss
import optimizers

np.random.seed(47)


class DenseNet:
    def __init__(self, input_dim, optim_config, loss_fn):
        self.graph = Graph(input_dim, optim_config, loss_fn)

    def addlayer(self, activation, units):
        return self.graph.addgate(activation, units)

    def train(self, X, Y):
        output = self.graph.forward(X)
        loss_val = self.graph.backward(Y)
        print("Loss :", loss_val)
        self.graph.update()
        return output, loss_val

    def predict(self, X):
        return self.graph.forward(X)


class Graph:
    def __init__(self, input_dim, optim_config, loss_fn):
        self.input_dim = input_dim
        self.network = list()
        self.loss_fn = loss_fn
        self.optim_config = optim_config
        self.predicted_output = None
        self.input = None

    def addgate(self, activation, units):
        if len(self.network) != 0:
            dim = self.network[-1].m
        else:
            dim = self.input_dim
        if activation == "Linear":
            self.network.append(Linear(dim, units))

    def forward(self, input_):
        self.input = input_
        output = None
        for layer in self.network:
            output = layer.forward(input_)
            input_ = output
        self.predicted_output = output
        return output

    def backward(self, expected):
        if self.loss_fn == "l2":
            loss_val, dz = loss.l2_loss(self.predicted_output, expected)
        elif self.loss_fn == "l1":
            loss_val, dz = loss.l1_loss(self.predicted_output, expected)
        for layer in reversed(self.network):
            dx, dw, db = layer.backward(dz)
            layer.dx, layer.dw, layer.db = dx, dw, db
            dz = dx
        return loss_val

    def update(self):
        if self.optim_config['type'] == 'sgd':
            for layer in self.network:
                layer.w, config = optimizers.sgd(layer.w, layer.dw, self.optim_config)
                layer.b, config = optimizers.sgd(layer.b, layer.db, self.optim_config)


class Linear:
    def __init__(self, d, m):
        self.m = m
        self.d = d
        self.out1, self.cache1 = None, None
        self.w = np.random.rand(d, m)
        self.b = np.random.rand(m)
        self.dw, self.dx, self.db = None, None, None

    def forward(self, input_):
        self.out1, self.cache1 = layers.linear_forward(input_, self.w, self.b)
        return self.out1

    def backward(self, dz):
        dx, dw, db = layers.linear_backward(dz, self.cache1)
        return dx, dw, db


dn = DenseNet(input_dim=2, optim_config={"type": "sgd", "learning_rate": 0.5}, loss_fn='l2')

dn.addlayer("Linear", 1)


# X = np.array([[0.1, 0.2], [0.1, 0.1],[0.2, 0.3]])
# Y = np.array([[0.3, 0.1], [0.2, 0.1], [0.5, 0.2]])

X = np.array([[0.1, 0.2], [0.1, 0.1],[0.2, 0.3]])
Y = np.array([[0.3], [0.2], [0.5]])

for i in range(1000):
    dn.train(X, Y)
print( "Ans is: ",dn.predict(np.array([[0.1, 0.6]])))

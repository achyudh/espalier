import layers, loss, optimizers
import numpy as np
from sklearn import datasets
from util import  to_categorical

class DenseNet:
    def __init__(self, input_dim, optim_config, loss_fn):
        self.graph = Graph(input_dim, optim_config, loss_fn)

    def addlayer(self, activation, units):
        return self.graph.addgate(activation, units)

    def train(self, X, Y):
        output = self.graph.forward(X)
        loss_val = self.graph.backward(Y)
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
        self.output = None

    def addgate(self, activation, units=0):
        if(len(self.network) != 0):
            dim = self.network[-1].m
        else:
            dim = self.input_dim
        if activation == "ReLU":
            self.network.append(ReLU(dim, units))
            return "SUCCESS: ReLU layer added"
        elif activation == "Linear":
            self.network.append(Linear(dim, units))
            return "SUCCESS: Linear layer added"
        # elif activation == "Softmax":
        #     self.network.append(Softmax(dim + 1, units))
        # elif activation == "Sigmoid":
        #     self.network.append(Sigmoid(dim + 1, units))
        else:
            return "ERROR: Unknown layer type"

    def forward(self, input):
        for layer in self.network:
            output = layer.forward(input)
            input = output
        self.output = output
        return output

    def backward(self, expected):
        if self.loss_fn == "svm":
            loss_val, dz = loss.svm_loss(self.output, expected)
        elif self.loss_fn == "softmax":
            loss_val, dz = loss.softmax_loss(self.output, expected)
        elif self.loss_fn == "l2":
            loss_val, dz = loss.l2_loss(self.output, expected)
        elif self.loss_fn == "l1":
            loss_val, dz = loss.l1_loss(self.output, expected)
        else:
            return "ERROR: Unknown loss function type"
        for layer in reversed(self.network):
            dx, dw, db = layer.backward(dz)
            layer.dx = dx
            layer.db = db
            layer.dw = dw
            dz = dx
        return loss_val

    def update(self):
        if self.optim_config['type'] == 'sgd':
            for layer in self.network:
                layer.w, config = optimizers.sgd(layer.w, layer.dw, self.optim_config)
        elif self.optim_config['type'] == 'sgd_momentum':
            layer_opt_config = self.optim_config
            for layer in self.network:
                layer.w, layer_opt_config = optimizers.sgd_momentum(layer.w, layer.dw, layer_opt_config)
        else:
            return "ERROR: Unknown optimizer type"


class ReLU:
    def __init__(self, d, m):
        self.m = m
        self.d = d
        self.out1, self.out2, self.cache1, self.cache2 = None, None, None, None
        self.w = np.random.rand(d, m)
        self.b = np.random.rand(m)
        self.dw, self.dx, self.db = None, None, None

    def forward(self, input):
        self.out1, self.cache1 = layers.linear_forward(input, self.w, self.b)
        self.out2, self.cache2 = layers.relu_forward(self.out1)
        return self.out2

    def backward(self, dz):
        dx1 = layers.relu_backward(dz, self.cache2)
        dx2, dw, db = layers.linear_backward(dx1, self.cache1)
        return dx2, dw, db


class Sigmoid:
    def __init__(self, d, m):
        self.m = m
        self.d = d
        self.out1, self.out2, self.cache1, self.cache2 = None, None, None, None
        self.w = np.random.rand(d, m)
        self.b = np.random.rand(m)
        self.dw, self.dx, self.db = None, None, None

    def forward(self, input):
        self.out1, self.cache1 = layers.linear_forward(input, self.w, self.b)
        self.out2, self.cache2 = layers.relu_forward(self.out1)
        return self.out2

    def backward(self, dz):
        dx1 = layers.relu_backward(dz, self.cache2)
        dx2, dw, db = layers.linear_backward(dx1, self.cache1)
        return dx2, dw, db


class Linear:
    def __init__(self, d, m):
        self.m = m
        self.d = d
        self.out1, self.cache1 = None, None
        self.w = np.random.rand(d, m)
        self.b = np.random.rand(m)

    def forward(self, input):
        self.out1, self.cache1 = layers.linear_forward(input, self.w, self.b)
        return self.out1

    def backward(self, dz):
        dx, dw, db = layers.linear_backward(dz, self.cache1)
        return dx, dw, db

iris = datasets.load_iris()
X = iris.data
Y = iris.target
Y = to_categorical(iris.target, 3)
dn = DenseNet(input_dim=4, optim_config={"type":"sgd", "learning_rate":0.5}, loss_fn='l2')
dn.addlayer("ReLU", 4)
dn.addlayer("ReLU", 3)
print(dn.train(X, Y, 1))
import numpy as np

import layers
import loss
import optimizers

from sklearn.datasets import load_iris
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
        elif activation == "ReLU":
            self.network.append(ReLU(dim, units))
        elif activation == "Sigmoid":
            self.network.append(Sigmoid(dim, units))

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
        elif self.loss_fn == "softmax":
            loss_val, dz = loss.softmax_loss(self.predicted_output, expected)
        elif self.loss_fn == "svm":
            loss_val, dz = loss.svm_loss(self.predicted_output, expected)
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
        elif self.optim_config['type'] == 'momentum':
            for layer in self.network:
                layer.w, config = optimizers.sgd_momentum(layer.w, layer.dw, self.optim_config)
                layer.b, config = optimizers.sgd_momentum(layer.b, layer.db, self.optim_config)



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


class ReLU:
    def __init__(self, d, m):
        self.m = m
        self.d = d
        self.out1, self.out2, self.cache1, self.cache2 = None, None, None, None
        self.w = 2 * np.random.rand(d, m) - 1
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
        self.w = 2*np.random.rand(d, m)-1
        self.b = np.random.rand(m)
        self.dw, self.dx, self.db = None, None, None

    def forward(self, input):
        self.out1, self.cache1 = layers.linear_forward(input, self.w, self.b)
        self.out2, self.cache2 = layers.sigmoid_forward(self.out1)
        return self.out2

    def backward(self, dz):
        dx1 = layers.sigmoid_backward(dz, self.cache2)
        dx2, dw, db = layers.linear_backward(dx1, self.cache1)
        return dx2, dw, db


def two_bit_xor_relu():
    print("Initializing net for two bit xor problem. . .")
    dn = DenseNet(input_dim=2, optim_config={"type": "sgd", "learning_rate": 0.3}, loss_fn='l2')

    dn.addlayer("ReLU", 2)
    dn.addlayer("ReLU", 1)

    X = np.array([[1.0, 1.0], [0.0, 1.0], [0.0, 0.0]])
    Y = np.array([[0.0], [1.0], [0.0]])

    for i in range(50):
        print("Iteration: ", i)
        dn.train(X, Y)
    print("Ans is: ", dn.predict(np.array([[1, 0]])))


def add_three_numbers():
    print("Initializing net for adding three numbers. . .")
    dn = DenseNet(input_dim=3, optim_config={"type": "sgd", "learning_rate": 0.5}, loss_fn='l2')

    dn.addlayer("Linear", 1)
    X = np.array([[0.1, 0.2, 0.4], [0.1, 0.1, 0.3], [0.2, 0.3, 0.6]])
    Y = np.array([[0.7], [0.5], [1.1]])

    for i in range(50):
        print("Iteration: ", i)
        dn.train(X, Y)
    print("Ans is: ", dn.predict(np.array([[0.2, 0.3, 0.1]])))


def two_bit_xor_sigmoid():
    print("Initializing net for two bit xor problem. . . ")
    dn = DenseNet(input_dim=2, optim_config={"type": "sgd", "learning_rate": 0.3}, loss_fn='l2')

    dn.addlayer("Sigmoid", 2)
    dn.addlayer("Sigmoid", 1)

    X = np.array([[1.0, 1.0], [0.0, 1.0], [0.0, 0.0]])
    Y = np.array([[0.0], [1.0], [0.0]])

    for i in range(300):
        print("Iteration: ", i)
        dn.train(X, Y)
    print("Ans is: ", dn.predict(np.array([[1, 0]])))

def iris_softmax():
    print("Initializing net for Iris dataset classification problem. . .")
    iris = load_iris()
    X = iris.data
    Y = iris.target

    dn = DenseNet(input_dim=4, optim_config={"type": "sgd", "learning_rate": 0.05}, loss_fn='softmax')
    dn.addlayer("ReLU", 4)
    dn.addlayer("ReLU", 6)
    dn.addlayer("ReLU", 3)

    for i in range(600):
        print("Iteration: ", i)
        dn.train(X, Y)

def iris_svm():
    print("Initializing net for Iris dataset classification problem. . .")
    iris = load_iris()
    X = iris.data
    Y = iris.target

    dn = DenseNet(input_dim=4, optim_config={"type": "sgd", "learning_rate": 0.01}, loss_fn='svm')
    dn.addlayer("ReLU", 4)
    dn.addlayer("ReLU", 6)
    dn.addlayer("ReLU", 3)

    for i in range(1000):
        print("Iteration: ", i)
        dn.train(X, Y)

# def iris_svm_momentum():
#     print("Initializing net for Iris dataset classification problem. . .")
#     iris = load_iris()
#     X = iris.data
#     Y = iris.target
#
#     dn = DenseNet(input_dim=4, optim_config={"type": "momentum", "learning_rate": 0.01, "momentum":0.5}, loss_fn='svm')
#     dn.addlayer("ReLU", 4)
#     dn.addlayer("ReLU", 6)
#     dn.addlayer("ReLU", 3)
#
#     for i in range(1000):
#         print("Iteration: ", i)
#         dn.train(X, Y)

#two_bit_xor_sigmoid()
print("*******************")
add_three_numbers()
# iris_svm_momentum()
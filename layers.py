import numpy as np


def linear_forward(x, w, b):
    N = x.shape[0]
    D = np.prod(x.shape[1:])
    x2 = np.reshape(x, (N, D))
    out = np.dot(x2, w) + b
    cache = (x, w, b)
    return out, cache


def linear_backward(dout, cache):
    x, w, b = cache
    dx = np.dot(dout, w.T).reshape(x.shape)
    dw = np.dot(x.reshape(x.shape[0], np.prod(x.shape[1:])).T, dout)
    db = np.sum(dout, axis=0)
    return dx, dw, db


def relu_forward(x):
    out = np.maximum(0, x)
    cache = x
    return out, cache


def relu_backward(dout, cache):
    x = cache
    dx = np.array(dout, copy=True)
    dx[x <= 0] = 0
    return dx




import mnist
import numpy as np


class Input_layer:
    X = mnist.download_and_parse_mnist_file("train-images-idx3-ubyte.gz")
    Y = mnist.download_and_parse_mnist_file("train-labels-idx1-ubyte.gz")
    idx = None

    def __init__(self, idx):
        self.idx = idx

    def getX(self):
        ls = self.X[self.idx].flatten() / 255
        try:
            return ls.reshape(len(self.idx), 28 * 28).T
        except TypeError:
            return ls.reshape(28 * 28, 1)

    def getY(self):
        return self.Y[self.idx]

    def onehot(self):
        ans_ls = self.Y[self.idx]
        C = 10
        B = len(self.idx)
        res = np.zeros((C, B))
        for i in range(B):
            res[ans_ls[i]][i] = 1
        return res

    def output(self):
        return self.getX()


class Test_input:
    X = mnist.download_and_parse_mnist_file("t10k-images-idx3-ubyte.gz")
    Y = mnist.download_and_parse_mnist_file("t10k-labels-idx1-ubyte.gz")
    idx = None

    def __init__(self, idx):
        self.idx = idx

    def getX(self):
        ls = self.X[self.idx].flatten() / 255
        try:
            return ls.reshape(len(self.idx), 28 * 28).T
        except TypeError:
            return ls.reshape(28 * 28, 1)

    def getY(self):
        return self.Y[self.idx]

    def output(self):
        return self.getX()


class Middle_layer:
    w1 = None
    x = None
    b1 = None

    def __init__(self, w1, x, b1, ReLU=False):
        self.w1 = w1
        self.x = x
        self.b1 = b1
        self.ReLU = ReLU

    def sigmoid(self, t):
        return 1 / (1 + np.exp(-t))

    def reLU(self, t):
        return np.maximum(0, t)

    def input(self):
        return np.dot(self.w1, self.x) + self.b1

    def output(self):
        if self.ReLU:
            return self.reLU(self.input())
        else:
            return self.sigmoid(self.input())


class Output_layer:
    w2 = None
    y = None
    b2 = None

    def __init__(self, w2, y, b2):
        self.w2 = w2
        self.y = y
        self.b2 = b2

    def softmax(self, a):
        alph = np.amax(a, axis=0, keepdims=True)
        t = a - alph
        return np.exp(t) / np.exp(t).sum(axis=0)[np.newaxis, :]

    def input(self):
        return self.w2 @ self.y + self.b2

    def output(self):
        return self.softmax(self.input())

    def ans(self):
        return self.output().argmax(axis=0)


class Cross_error:
    ans = None
    out = None
    size = None

    def __init__(self, ans, out, size):
        self.ans = ans
        self.out = out
        self.size = size

    def cross_error(self):
        return np.sum(-self.ans * np.log(self.out)) / self.size

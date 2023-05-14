import numpy as np


class Back_prop:
    params = {}

    def __init__(self, lay1, lay2, lay3, size, lr, Momentum=False, alpha=0.9, dw1=0, db1=0, dw2=0, db2=0):
        self.lay1 = lay1
        self.lay2 = lay2
        self.lay3 = lay3
        self.size = size
        self.lr = lr
        self.Momentum = Momentum
        self.alpha = alpha
        self.dw1 = dw1
        self.db1 = db1
        self.dw2 = dw2
        self.db2 = db2
        self.params["En"] = self.bp_err()
        self.params = self.params | {
            "w2_err": self.grad_w(En=self.params["En"], X=self.lay2.output()),
            "b2_err": self.grad_b(En=self.params["En"]),
            "x2_err": self.grad_x(W=self.lay3.w2, En=self.params["En"]),
        }
        if self.lay2.ReLU:
            self.params = self.params | {
                "En2": self.grad_ReLU(En=self.params["x2_err"], Y=self.lay2.output()),
            }
        else:
            self.params = self.params | {
                "En2": self.grad_sig(En=self.params["x2_err"], Y=self.lay2.output()),
            }
        self.params = self.params | {
            "w1_err": self.grad_w(En=self.params["En2"], X=self.lay1.output()),
            "b1_err": self.grad_b(En=self.params["En2"]),
        }

    def bp_err(self):
        return (self.lay3.output() - self.lay1.onehot()) / self.size

    def grad_w(self, En, X):
        return np.dot(En, X.T)

    def grad_b(self, En):
        return np.sum(En, axis=1, keepdims=True)

    def grad_x(self, W, En):
        return np.dot(W.T, En)

    def grad_sig(self, En, Y):
        return En * (1 - Y) * Y

    def grad_ReLU(self, En, Y):
        return En * (np.where(Y > 0, 1, 0))

    def update(self):
        if self.Momentum:
            self.dw1 = self.alpha * self.dw1 - self.lr * self.params["w1_err"]
            self.db1 = self.alpha * self.db1 - self.lr * self.params["b1_err"]
            self.dw2 = self.alpha * self.dw2 - self.lr * self.params["w2_err"]
            self.db2 = self.alpha * self.db2 - self.lr * self.params["b2_err"]
            return {
                "w1": self.lay2.w1 + self.dw1,
                "b1": self.lay2.b1 + self.db1,
                "w2": self.lay3.w2 + self.dw2,
                "b2": self.lay3.b2 + self.db2,
            }
        else:
            return {
                "w1": self.lay2.w1 - self.lr * self.params["w1_err"],
                "b1": self.lay2.b1 - self.lr * self.params["b1_err"],
                "w2": self.lay3.w2 - self.lr * self.params["w2_err"],
                "b2": self.lay3.b2 - self.lr * self.params["b2_err"],
            }

    def momentum(self):
        return {
            "dw1": self.dw1,
            "db1": self.db1,
            "dw2": self.dw2,
            "db2": self.db2,
        }

    def accuracy(self):
        return sum(self.lay1.getY()[i] == self.lay3.ans()[i] for i in range(self.size)) / self.size

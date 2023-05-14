from layer import Input_layer, Middle_layer, Output_layer, Cross_error
from bp import Back_prop
import numpy as np

# set size of parameters
D = 28 * 28
C = 10
M = 100
B = 100
LEARNING_RATE = 0.01
COUNT = 60000
THRESHOLD = 10 ** (-4)

# generate weights and biases
np.random.seed(seed=777)
w1 = np.random.normal(loc=0, scale=1 / np.sqrt(D), size=(M, D))
b1 = np.random.normal(loc=0, scale=1 / np.sqrt(D), size=(M, 1))
w2 = np.random.normal(loc=0, scale=1 / np.sqrt(M), size=(C, M))
b2 = np.random.normal(loc=0, scale=1 / np.sqrt(M), size=(C, 1))

# generate batch and variables (to calculate delta_En)
batch = list(range(COUNT))  # index_array [0,1,2,...,59999]
dE = 100
tmp = 100

# until delta_En converges
while THRESHOLD < dE:
    # shuffle batch list
    np.random.shuffle(batch)
    e_sum = 0

    # learn each mini-batch and update the values of weights and biases
    for mini in np.array_split(batch, int(COUNT / B)):
        lay1 = Input_layer(idx=mini)
        lay2 = Middle_layer(w1=w1, x=lay1.output(), b1=b1)
        lay3 = Output_layer(w2=w2, y=lay2.output(), b2=b2)
        e = Cross_error(ans=lay1.onehot(), out=lay3.output(), size=B)
        e_sum += e.cross_error()
        bp = Back_prop(lay1=lay1, lay2=lay2, lay3=lay3, size=B, lr=LEARNING_RATE)
        w1, b1, w2, b2 = bp.update().values()

    # calculate delta_En
    dE = tmp - e_sum / (COUNT / B)
    tmp = e_sum / (COUNT / B)

    # print result in the epoch
    print("cross_error: ", e_sum / (COUNT / B))
    print("delta_En: ", dE)

# save learned weights and biases
np.savez("result", w1=w1, b1=b1, w2=w2, b2=b2)
print("end")

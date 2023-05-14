import numpy as np
from layer import Middle_layer, Output_layer, Test_input

# load weights and biases
w1, b1, w2, b2 = np.load("DNN-A2Z/result.npz").values()

l1 = Test_input(idx=int(input()))
l2 = Middle_layer(w1=w1, x=l1.output(), b1=b1)
l3 = Output_layer(w2=w2, y=l2.output(), b2=b2)

print("Output answer: ", l3.ans()[0])
print("Correct answer: ", l1.getY())

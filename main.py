#    Let's build a really simple neural network from scratch
#    Our goal is to train the network to imitate this function:
#
#        f(x) = 2 * abs(3 * x + 3) + 2
#
#    We will use this structure:
#
#
#                ReLU
#            t1 ------> h1
#           / /           \
#      W1  / /         W2  \
#         / /    ReLU       \
#    input - t2 ------> h2 - output
#         / /               /
#         B1               B2
#
#
def relu(x):
    return np.maximum(x, 0)

def d_relu(x):
    return (x >= 0).astype(float)

import numpy as np

#random initialization
b1 = np.random.random((2, 1))
w1 = np.random.random((2, 1))
b2 = np.random.random((1, 1))
w2 = np.random.random((2, 1))

alpha = 0.01

list_err = []

#test

print('Before learning:')
print()

for i in range(1):
    input = np.random.random((1, 1))
    y = 2 * abs(3 * input + 3) + 2

    t1 = w1 @ input + b1
    h1 = relu(t1)
    output = w2.T @ h1 + b2

    E = (output - y) ** 2

    print('Neural network output: ' + str(output))
    print('Correct output: ' + str(y))
    print('Error: ' + str(E))

print()
print()

for i in range(100000):

    input = np.random.random((1, 1))
    y = 2 * abs(3 * input + 3) + 2

    #forward
    t1 = w1 @ input + b1
    h1 = relu(t1)
    output = w2.T @ h1 + b2

    #backward
    E = (output - y) ** 2
    dE_dout = 2 * (output - y)
    dE_dt2 = dE_dout
    dE_dw2 = h1 @ dE_dt2
    dE_db2 = dE_dt2
    dE_dh1 = dE_dt2 @ w2.T
    dE_dt1 = dE_dh1 @ d_relu(t1)
    dE_dw1 = input @ dE_dt1
    dE_db1 = dE_dt1

    #update
    w2 = w2 - dE_dw2 * alpha
    b2 = b2 - dE_db2 * alpha
    w1 = w1 - dE_dw1 * alpha
    b1 = b1 - dE_db1 * alpha

    #add error
    list_err.append(E[0][0])

#test

print('After learning:')
print()

for i in range(1):
    input = np.random.random((1, 1))
    y = 2 * abs(3 * input + 3) + 2

    t1 = w1 @ input + b1
    h1 = relu(t1)
    output = w2.T @ h1 + b2

    E = (output - y) ** 2

    print('Neural network output: ' + str(output))
    print('Correct output: ' + str(y))
    print('Error: ' + str(E))

import matplotlib.pyplot as plt
plt.plot(list_err)
plt.show()

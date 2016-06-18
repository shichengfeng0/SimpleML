from __future__ import division
from numpy import *
import scipy.sparse as sp
from copy import copy
import matplotlib.pyplot as plt

training_data = genfromtxt('train.txt', delimiter=',')
Y_train = training_data[:,0]
X_train = training_data[:, 1:]
Y_test = genfromtxt('test_label.txt', delimiter=',')
X_test = genfromtxt('test.txt', delimiter=',')
oversampled_data = genfromtxt('oversampled_train.txt', delimiter=',')
X_oversampled_train = oversampled_data[:, 1:]
Y_oversampled_train = oversampled_data[:, 0]

def computeW(x_measure, y_measure, steps):
    num_feasure = len(x_measure[0])
    w = [0] * (num_feasure + 1) # additional one for the bias term
    num_examples = len(x_measure)

    for t in range(0, steps):
        for i in range(0, num_examples):
            curX = x_measure[i]
            curY = y_measure[i]
            measured_val = dot(w[1:], curX.T) + w[0]
            measured_sign = 1

            if (measured_val <= 0):
                measured_sign = -1

            if (measured_sign == 1 and curY == 0):
                w[1:] = w[1:] - curX
                w[0] = w[0] - 1
            elif (measured_sign == -1 and curY == 1):
                w[1:] = w[1:] + curX
                w[0] = w[0] + 1

    return w

def predictPrecisionAndRecall(x_measure, y_measure, w):
    actualpos = 0
    labeledpos = 0
    truepos = 0
    trueneg = 0
    num_sample = len(x_measure)

    for i in range(0, num_sample):
        measured_Y = dot(w[1:], x_measure[i]) + w[0]
        measured_sign = 1
        if (measured_Y <= 0):
            measured_sign = -1

        actual_Y = y_measure[i]

        if (actual_Y == 1):
            actualpos += 1
            if (measured_sign == 1):
                truepos += 1
        else: # actual_y = 0
            if (measured_sign == -1):
                trueneg += 1

        if (measured_sign == 1):
            labeledpos += 1

    print("actual positive {0}, labeled positive is {1}, true positive is {2}\n".format(actualpos, labeledpos, truepos))
    print("actual negative {0}, labeled negative is {1}, true negative is {2}\n".format(num_sample - actualpos, num_sample - labeledpos, trueneg))
    print("for class 1, precision is {0}, recall is {1}\n".format((truepos / labeledpos), (truepos / actualpos)))
    print("for class 0, precision is {0}, recall is {1}\n".format((trueneg / (num_sample - labeledpos)), trueneg / (num_sample - actualpos)))

def validation(train_data):
    random.shuffle(train_data)
    y_measured = train_data[:,0]
    x_measured = train_data[:, 1:]

    cut = 10000;
    w = computeW(x_measured[0:cut, :], y_measured[0:cut], 50)

    predictPrecisionAndRecall(x_measured[cut:, :], y_measured[cut:], w)


# w = computeW(X_oversampled_train, Y_oversampled_train, 50)
# predictPrecisionAndRecall(X_test, Y_test, w)

validation(oversampled_data)



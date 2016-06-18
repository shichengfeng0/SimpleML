#!/usr/bin/python
from __future__ import division
from numpy import *
import scipy.sparse as sp
from copy import copy
import matplotlib.pyplot as plt

# import the data
training_data = genfromtxt('train.txt', delimiter=',')
Y_train = training_data[:,0]
X_train = training_data[:, 1:]
Y_test = genfromtxt('test_label.txt', delimiter=',')
X_test = genfromtxt('test.txt', delimiter=',')
oversampled_data = genfromtxt('oversampled_train.txt', delimiter=',')
X_oversampled_train = oversampled_data[:, 1:]
Y_oversampled_train = oversampled_data[:, 0]

AllOne = [1] * len(X_train)
AllOneT = sp.csc_matrix(AllOne).transpose()
X_train = sp.hstack([AllOneT, X_train])
Y_train = sp.csc_matrix(Y_train)

AllOne = [1] * len(X_oversampled_train)
AllOneT = sp.csc_matrix(AllOne).transpose()
X_oversampled_train = sp.hstack([AllOneT, X_oversampled_train])
Y_oversampled_train = sp.csc_matrix(Y_oversampled_train)

AllOne = [1] * len(X_test)
AllOneT = sp.csc_matrix(AllOne).transpose()
X_test = sp.hstack([AllOneT, X_test])
Y_test = sp.csc_matrix(Y_test)


# this method will return 1 * 10000 np 2D array
def computeProOf1(x_total, w):
    line = w * x_total.transpose().todense() # 1 * 10000
    e = exp(line) # 1 * 10000
    allOne = [1] * len(e)
    return e / (e + allOne)

def computeW(x_total, y_total, lmd, lr, steps):
    # N = len(x_total[0])
    N = x_total.shape[0]
    w = sp.csc_matrix([0] * 55) # 1 * 55

    lossArray = [];

    for i in range(0, steps):
        w0 = w[0,0]
        x_total0 = (x_total.todense())[:, :1]
        w0 = w0 - lr * (-y_total * x_total0 / N + computeProOf1(x_total, w) * x_total * 1.0 / N)
        w = w - lr * (lmd * w - y_total * x_total * 1.0 / N + computeProOf1(x_total, w) * x_total * 1.0 / N)
        w[0,0] = w0[0,0]
        lossValue = computeLossFunc(x_total, y_total, w, lmd)
        # print(lossValue)
        lossArray.append(lossValue)

    plt.plot(lossArray)
    plt.show()
    return w

def computeLossFunc(x_total, y_total, w, lmd):
    result = 0
    N = 10000

    w0 = w[0, 0]
    firstP = ((w * w.transpose())[0,0] - w0 * w0) * lmd * 0.5
    line = w * x_total.transpose() # 1 * 10000
    e = exp(line)
    allOne = [1] * len(e)
    ln = log(e + allOne) # 1 * 10000
    dotV = dot(y_total.todense(), sp.csc_matrix(line).transpose().todense())
    secP = (sum(dotV) - sum(ln)) / N

    return firstP - secP;

def predictAccuracy(x_measure, y_measure, w):
    correct = 0
    result = computeProOf1(x_measure, w)
    print("the predict CTRs is ")
    print(w)

    for i in range(0, x_measure.shape[0]):
        curRes = result[0, i]
        if (curRes >= 0.5 and y_measure[0, i] == 1):
            correct = correct + 1
        elif(curRes < 0.5 and y_measure[0, i] == 0):
            correct = correct + 1

    print("the accuracy of the test data is {0}\n".format(correct * 1.0 / x_measure.shape[0]))

def computeWStopWhenLessDecrease(x_total, y_total, lmd, lr, ebs):
    N = x_total.shape[0]
    w = sp.csc_matrix([0] * 55) # 1 * 55
    lossArray = []

    # get the first loss function value
    w0 = w[0,0]
    x_total0 = (x_total.todense())[:, :1]
    w0 = w0 - lr * (-y_total * x_total0 / N + computeProOf1(x_total, w) * x_total * 1.0 / N)
    w = w - lr * (lmd * w - y_total * x_total * 1.0 / N + computeProOf1(x_total, w) * x_total * 1.0 / N)
    w[0,0] = w0[0,0]
    preLossValue = computeLossFunc(x_total, y_total, w, lmd)
    lossArray.append(preLossValue)

    time = 1
    while (True):
        time = time + 1
        w0 = w[0,0]
        x_total0 = (x_total.todense())[:, :1]
        w0 = w0 - lr * (-y_total * x_total0 / N + computeProOf1(x_total, w) * x_total * 1.0 / N)
        w = w - lr * (lmd * w - y_total * x_total * 1.0 / N + computeProOf1(x_total, w) * x_total * 1.0 / N)
        w[0,0] = w0[0,0]
        curLossValue = computeLossFunc(x_total, y_total, w, lmd)
        lossArray.append(curLossValue)
        if (abs(preLossValue - curLossValue) < ebs):
            plt.plot(lossArray)
            plt.show()
            print("it takes {0} times to get the loss value to change less than {1}\n".format(time, ebs))
            return w
        else:
            preLossValue = curLossValue

def computeL2(w):
    return sqrt((w * w.transpose())[0,0])

def predictPrecisionAndRecall(x_measure, y_measure, w):
    correct = 0
    result = w * x_measure.transpose()
    print w

    truepos = 0;
    labeledpos = 0
    actualpos = 0
    trueneg = 0

    num_sample = x_measure.shape[0]

    for i in range(0, num_sample):
        curRes = result[0, i]
        if (y_measure[0, i] == 1):
            actualpos = actualpos + 1

        if (curRes > 0):
            labeledpos = labeledpos + 1
            if (y_measure[0, i] == 1):
                truepos = truepos + 1
        else:
            if (y_measure[0, i] == 0):
                trueneg = trueneg + 1

    print("the ture positive is {0}, the labeled positive is {1}, the actual positive is {2}".format(truepos, labeledpos, actualpos))
    print("the ture negative is {0}, the labeled negative is {1}, the actual negative is {2}".format(trueneg, num_sample - labeledpos, num_sample - actualpos))
    # print("for class 1, precision is {0}, recall is {1}\n".format((truepos / labeledpos), (truepos / actualpos)))
    # print("for class 0, precision is {0}, recall is {1}\n".format((trueneg / (num_sample - labeledpos)), trueneg / (num_sample - actualpos)))

# part 2 a
# w1 = computeW(X_train, Y_train, 0.3, 0.1, 1000)
# print(w1)

# part 2 b
# predictAccuracy(X_test, Y_test, w1)

# part 3 a,b
# w2 = computeWStopWhenLessDecrease(X_train, Y_train, 0.3, 0.1, 0.0005)

# part 3 c
# predictAccuracy(X_test, Y_test, w2)

# part 4
# w3 = computeW(X_train, Y_train, 0, 0.1, 1000)
# w3 = w3[:,1:]
# l21 = computeL2(w3)
# print("L2 norm is {0}, when lmd = 0".format(l21))
#
# w4 = computeW(X_train, Y_train, 0.3, 0.1, 1000)
# w4 = w4[:,1:]
# l22 = computeL2(w4)
# print ("L2 norm is {0}, when lmd = 0.3".format(l22))
# lambda is 0.3 is preferred since the l2 norm is much smaller when lambda 0


# part 5 a
# w5 = computeW(X_train, Y_train, 0.3, 0.01, 5000)
# predictPrecisionAndRecall(X_test, Y_test, w5)

# part 5 b
# w6 = computeW(X_oversampled_train, Y_oversampled_train, 0.3, 0.01, 5000)
# predictPrecisionAndRecall(X_test, Y_test, w6)

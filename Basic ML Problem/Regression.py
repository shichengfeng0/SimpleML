# Chengfeng Shi
# 1237783
# CSE446

import numpy as np
import scipy.sparse as sp
import math as math
import matplotlib.pyplot as plt


# Load a text file of integers:
y = np.loadtxt("upvote_labels.txt", dtype=np.int)
# Load a text file of strings:
featureNames = open("upvote_features_100.txt").read().splitlines()
# Load a csv of floats:
A = sp.csc_matrix(np.genfromtxt("upvote_data_100.csv", delimiter=","))


AllOne = [1] * 6000
AllOneT = sp.csc_matrix(AllOne).transpose()

B = sp.hstack([AllOneT, A]).todense();

# H is the training set with size 5000
H = B[:5000, :]
HT = H.transpose()
Y = sp.csc_matrix(y).transpose().todense();
Ytrain = Y[:5000, :]


# part 1
w = (np.linalg.pinv(HT * H)) * HT * Ytrain
# print w

# compute the error
# training set error
Etrain = (H * w - Ytrain).transpose() * (H * w - Ytrain)
SqrtEtrain = math.sqrt(Etrain / 5000)
print("training error is {0}".format(SqrtEtrain))

# test set error
Htest = B[5000:, :];
Ytest = Y[5000:, :]
Etest = (Htest * w - Ytest).transpose() * (Htest * w - Ytest)
SqrtEtest = math.sqrt(Etest / 1000)
print("test error is {0}".format(SqrtEtest))

# used to split the training set into array HArray and YArray
def constructCrossData(k, Hf, Yf, HArray, YArray):
    interval = len(Hf) / k;
    for i in range(0, k):
        HArray.append(Hf[i * interval:(i + 1) * interval, :])
        YArray.append(Yf[i * interval:(i + 1) * interval, :])

# compute the w for the given training set
def computeWbridge(hf, yf, lf, imf):
    return np.linalg.pinv(hf.transpose() * hf + lf * imf) * hf.transpose() * yf

# compute the training error square
def computeErrorSqr(hf, yf, wf):
    error = (hf * wf - yf).transpose() * (hf * wf - yf)
    return error[0,0]

# compute the RMSE
def computeError(hf, yf, wf):
    errorSqr = computeErrorSqr(hf, yf, wf)
    return math.sqrt(errorSqr / len(hf));

# generate the identity matrix with the 0 in the upper left corner
def generateSpecialIM(size):
    im = np.identity(size);
    im[0, 0] = 0
    return im

# compute the weight and error
def computerBAndE(hf, yf, lf, imf):
    wb = computeWbridge(hf, yf, lf, imf)
    return computeError(hf, yf, wb)

# slice out the data in the specified range
def getRidOfDataInRange(data, start, end):
    part1 = data[:start, :]
    part2 = data[end:, :]

    if (len(part1) == 0):
        return part2
    elif (len(part2) == 0):
        return part1
    else:
        return sp.vstack([sp.csc_matrix(part1), sp.csc_matrix(part2)]).todense()

# compute the RMSE
def computeRMSE(Hdata, Ydata, lf):
    imf = generateSpecialIM(101);
    wb = computeWbridge(Hdata, Ydata, lf, imf)
    return computeError(Hdata, Ydata, wb)

# using the given w to compute the RMSE error in test data
def computeRMSEInTest(Hdata, Ydata, w):
    return computeError(Hdata, Ydata, w)

# compute the cross validate error for the given fold size
def computeCrossError(Hdata, Ydata, foldSize, lf):
    HArray = [];
    YArray = [];
    constructCrossData(foldSize, Hdata, Ydata, HArray, YArray)
    interval = len(Hdata) / foldSize
    crossErrorTotal = 0

    for i in range(0, foldSize):
        curTrainData = getRidOfDataInRange(Hdata, interval * i, interval * (i + 1))
        curTrainLabel = getRidOfDataInRange(Ydata, interval * i, interval * (i + 1))
        identityMatrix = generateSpecialIM(101)
        curW = computeWbridge(curTrainData, curTrainLabel, lf, identityMatrix)
        crossErrorTotal += computeError(HArray[i], YArray[i], curW)

    return crossErrorTotal / foldSize;


# method to run problem 4_2 and 4_3
def execute(foldSize):
    lmd = 1;
    bestLmd = 0
    bestCrossE = 1000000
    lmdArray = []
    RMSEArray = []
    crossEArray = []
    for j in range(0, 20):
        crossE = computeCrossError(H, Ytrain, foldSize, lmd)
        if (bestCrossE > crossE):
            bestLmd = lmd
            bestCrossE = crossE

        RMSE = computeRMSE(H, Ytrain, lmd)
        message = "with lambd {0}, cross error = {1}, RMSE = {2}".format(lmd, crossE, RMSE)
        print message
        crossEArray.append(crossE)
        RMSEArray.append(RMSE)
        lmdArray.append(lmd)

        lmd = lmd * 0.75

    print "the optimal lamda is {0}".format(bestLmd)

    # generate the graph
    plt.plot(lmdArray, RMSEArray)
    plt.plot(lmdArray, crossEArray)
    plt.show()

    # get the corresponding w for the best lmd of training set

    imf = generateSpecialIM(101)
    bestW = computeWbridge(H, Ytrain, bestLmd, imf)

    # compute the RMSE in test set using the predicted w
    RMSEtmp = computeRMSEInTest(B[5000:,:], Y[5000:,:], bestW)
    print "test RMSE is {0}".format(RMSEtmp)

####################### part 2 ##################################
execute(5)

######################## part 3 #################################
execute(10)

######################## part 4 #################################

# used to compute the validate error
def computeValidateError(Hdata, Ydata, lf):
    newTrainData = getRidOfDataInRange(Hdata, 4000, 5000)
    newTrainResult = getRidOfDataInRange(Ydata, 4000, 5000)
    newTestData = getRidOfDataInRange(Hdata, 0, 4000)
    newTestResult = getRidOfDataInRange(Ydata, 0, 4000)

    imf = generateSpecialIM(101)
    w = computeWbridge(newTrainData, newTrainResult, lf, imf)
    return computeError(newTestData, newTestResult, w)

# used to calculate problem in part 4
def part4():
    lmd = 1;

    bestLmd = 0
    bestValidateE = 1000000
    lmdArray = []
    RMSEArray = []
    validateEArray = []
    for j in range(0, 20):
        validateE = computeValidateError(H, Ytrain, lmd)
        if (bestValidateE > validateE):
            bestLmd = lmd
            bestValidateE = validateE

        RMSE = computeRMSE(H, Ytrain, lmd)
        message = "with lambd {0}, validate error = {1}, RMSE = {2}".format(lmd, validateE, RMSE)
        print message
        validateEArray.append(validateE)
        RMSEArray.append(RMSE)
        lmdArray.append(lmd)

        lmd = lmd * 0.75

    print "the optimal lamda is {0}".format(bestLmd)

    # generate the graph
    plt.plot(lmdArray, RMSEArray)
    plt.plot(lmdArray, validateEArray)
    plt.show()

    # get the corresponding w for the best lmd of training set

    imf = generateSpecialIM(101)
    bestW = computeWbridge(H, Ytrain, bestLmd, imf)

    # compute the RMSE in test set using the predicted w
    RMSEtmp = computeRMSEInTest(B[5000:,:], Y[5000:,:], bestW)
    print "test RMSE is {0}".format(RMSEtmp)

part4()
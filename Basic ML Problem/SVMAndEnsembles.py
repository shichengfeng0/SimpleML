from numpy import *
from sklearn.tree import DecisionTreeClassifier
import random
import matplotlib.pyplot as plt
import numpy as np

def load_data(filename):
    # result = []
    x_data = []
    y_data = []
    with open(filename) as f:
        for line in f:
            curData = [0] * 57

            for i in range(0, 57):
                c = line[i]
                if (c == 'a'):
                    curData[i] = 0
                elif (c == 't'):
                    curData[i] = 1
                elif (c == 'c'):
                    curData[i] = 2
                else :
                    curData[i] = 3

            x_data.append(curData)

            if (line[58] == '+'):
                y_data.append(1)
            else :
                y_data.append(-1)

    return (array(x_data), array(y_data))

(X_training, Y_training) = load_data("training.txt")
(X_test, Y_test) = load_data("test.txt")

def get_stumps_tree():
    clf = DecisionTreeClassifier(criterion = "entropy", max_depth = 1)
    return clf

def get_depth_two_tree():
    clf = DecisionTreeClassifier(criterion = "entropy", max_depth = 2)
    return clf

# return (x_random, y_corresponding)
def get_random_data_with_same_size_as_training_data(X_total, Y_total):
    x_result = []
    y_result = []
    size = X_total.shape[0]
    for i in range(0, size):
        random_index = random.randrange(0, size)
        x_result.append(X_total[random_index, :])
        y_result.append(Y_total[random_index])
    return (array(x_result), y_result)

# return the number of wrong predict
def compute_test_error_bagging(all_hypo, X_test, Y_test):
    result = [0] * Y_test.shape[0]
    for i in range(0, len(all_hypo)):
        result += all_hypo[i].predict(X_test)

    num_wrong = 0
    for i in range(0, len(Y_test)):
        if (result[i] > 0 and Y_test[i] == -1):
            num_wrong += 1
        elif (result[i] <= 0 and Y_test[i] == 1):
            num_wrong += 1

    return num_wrong


# t is the number of iteration
# return the wrong number of prediction of the given test
def bagging(type, X_training, Y_training, X_test, Y_test, t):
    all_wrong_num = [0] * t
    for j in range(0, 10):
        all_hypo = []
        time = []
        num_wrong = []

        for i in range(0, t):
            cur_clf = get_depth_two_tree()
            if (type == "stump_tree"):
                cur_clf = get_stumps_tree()

            (xx, yy) = get_random_data_with_same_size_as_training_data(X_training, Y_training)
            cur_clf.fit(xx, yy)
            all_hypo.append(cur_clf)
            num_wrong.append(compute_test_error_bagging(all_hypo, X_test, Y_test))

        all_wrong_num = np.add(all_wrong_num, num_wrong)


    time = []
    for i in range(1, t + 1):
        time.append(i)
    all_wrong_num = all_wrong_num / 10.0

    all_wrong_num = all_wrong_num * 1.0 / len(Y_test)

    plt.plot(time, all_wrong_num)
    plt.show()

def calculate_unique(input_data):
    total_ratio = 0
    all_unique = []
    time = []
    for j in range(0, 100):
        size = input_data.shape[0]
        seen = [0] * size
        unique = 0

        for i in range(0, size):
            random_index = random.randrange(0, size)
            if (seen[random_index] == 0):
                unique += 1
                seen[random_index] = 1

        all_unique.append(unique * 1.0 / size)
        time.append(j + 1)
        total_ratio += unique * 1.0 / size

    plt.plot(time, all_unique)
    plt.show()
    print("average ratio of the uniquenes is {0}".format(total_ratio / 100))

def get_weighted_stump_tree(X_training, Y_training, weight):
    clf = DecisionTreeClassifier(criterion = "entropy", max_depth = 1)
    clf.fit(X_training, Y_training, sample_weight=weight)
    return clf

def get_weighted_depth_two_tree(X_training, Y_training, weight):
    clf = DecisionTreeClassifier(criterion = "entropy", max_depth = 2)
    clf.fit(X_training, Y_training, sample_weight=weight)
    return clf

# according to the percent that each D weighted, generate a new D array that sum to 1
def regenerateD(D_percent):
    sum = 0.0
    for i in range(0, len(D_percent)):
        sum += D_percent[i]
    new_D = []

    for i in range(0, len(D_percent)):
        new_D.append(D_percent[i] * 1.0 / sum)

    return new_D

# return the number of wrong predict
def compute_test_error_adaboost(all_clf, all_a, X_test, Y_test):
    result = [0] * len(Y_test)
    for i in range(0, len(all_clf)):
        result += all_clf[i].predict(X_test) * all_a[i]

    num_wrong = 0
    for i in range(0, len(Y_test)):
        if (result[i] > 0 and Y_test[i] == -1):
            num_wrong += 1
        elif (result[i] <= 0 and Y_test[i] == 1):
            num_wrong += 1

    return num_wrong

def adaboost(type, X_training, Y_training, X_test, Y_test, t):
    D = [1.0 / X_training.shape[0]] * X_training.shape[0]
    all_a = []
    all_clf = []
    time = []
    all_wrong_num = []
    training_error = []

    for i in range(0, t):
        #  get the decision tree
        clf = get_weighted_stump_tree(X_training, Y_training, D)
        if (type == "depth_two_tree"):
            clf = get_weighted_depth_two_tree(X_training, Y_training, D)
        all_clf.append(clf)

        Y_predicted = clf.predict(X_training)

        e = 0
        for j in range(0, len(D)):
            if (Y_predicted[j] != Y_training[j]):
                e += D[j]

        if (e == 0):
            print("current the e is 0")
        cur_a = 0.5 * np.log((1 - e) / e)
        all_a.append(cur_a)

        time.append(i + 1)
        all_wrong_num.append(compute_test_error_adaboost(all_clf, all_a, X_test, Y_test))
        training_error.append(compute_test_error_adaboost(all_clf, all_a, X_training, Y_training))

        eayh = np.exp((Y_training * Y_predicted) * (-1) * cur_a)
        D = regenerateD(eayh * D)

    all_wrong_num = np.array(all_wrong_num) * 1.0 / len(Y_test)
    training_error = np.array(training_error) * 1.0 / len(Y_training)

    plt.plot(time, all_wrong_num)
    plt.plot(time, training_error)
    plt.show()


# 2.2.1 using stump tree
# bagging("stump_tree", X_training, Y_training, X_test, Y_test, 100)

# 2.2.2 using depth two tree
# bagging("depth_two_tree", X_training, Y_training, X_test, Y_test, 100)

# 2.2.3 calculate uniqueness
# calculate_unique(X_training)

# 2.3.1-stump_tree
# adaboost("stump_tree", X_training, Y_training, X_test, Y_test, 500)
# 2.3.1-depth_two_tree
# adaboost("depth_two_tree", X_training, Y_training, X_test, Y_test, 500)

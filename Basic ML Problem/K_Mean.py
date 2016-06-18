from numpy import *
import random
import matplotlib.pyplot as plt
import numpy as np
import sys


X_training = genfromtxt("digit.txt")
Y_training = genfromtxt("labels.txt", dtype=int)

def generate_start_centers_first_k(X_training, k):
    result = []
    for i in range(0, k):
        result.append(X_training[i])
    return result

def generate_start_centers_random(X_training, k):
    result = []
    for i in range(0, k):
        random_index = random.randrange(0, X_training.shape[0])
        result.append(X_training[random_index])
    return result

def compute_dist_sqr(x1, x2):
    # result = 0
    # for i in range(0, len(x1)):
    #     result += (x1[i] - x2[i]) *  (x1[i] - x2[i])
    # return result

    x1 = np.matrix(x1)
    x2 = np.matrix(x2)
    return ((x1 - x2) * (x1 - x2).T)[0,0]

def find_close_centers(X_training, all_centers):
    close_centers = []

    for i in range(0, X_training.shape[0]):
        cluster = -1
        cur_x = X_training[i]
        min_dist_sqr = sys.maxint
        for j in range(0, all_centers.shape[0]):
            cur_center = all_centers[j]

            cur_dist_sqr = compute_dist_sqr(cur_x, cur_center)

            if (cur_dist_sqr < min_dist_sqr):
                cluster = j
                min_dist_sqr = cur_dist_sqr

        close_centers.append(cluster)

    return close_centers

# recompute the centers according to the instance belongs to them
def recompute_center(belong_to, X_training, k):
    result = [[0.0] * X_training.shape[1]] * k
    result = np.matrix(result)
    num = [0.0] * k

    for i in range(0, X_training.shape[0]):
        index = belong_to[i]
        result[index] += X_training[i]
        num[index] += 1

    for i in range(0, k):
        # print("before")
        # print("divide by {0}".format(num[i]))
        # print(result[i])
        result[i] = np.array(result[i]) * 1.0 / num[i]
        # print("after")
        # print(result[i])


    # print(result)

    return result

def compute_sum_sqr(belong_to, all_centers, X_training):
    sum_sqrt = 0
    for i in range(0, X_training.shape[0]):
        cur_x = X_training[i]
        belong_to_center = all_centers[belong_to[i]]
        sum_sqrt += compute_dist_sqr(cur_x, belong_to_center)
    return sum_sqrt

def compute_total_mistake_rate(belong_to, X_training, Y_training, k):
    size = 4 # the number of the different results
    predict_result = [[0] * size] * k
    predict_result = np.matrix(predict_result)

    for i in range(0, X_training.shape[0]):
        cluster_index = belong_to[i]
        y_index = Y_training[i] / 2
        predict_result[cluster_index, y_index] += 1

    cluster_to_digit = []
    for i in range(0, k):
        cur_result = -1
        cur_num_vote = -1
        for j in range(0, size):
            if (predict_result[i,j] > cur_num_vote):
                cur_result = j * 2 + 1
                cur_num_vote = predict_result[i,j]

        cluster_to_digit.append(cur_result)

    wrong_predict = 0
    for i in range(0, X_training.shape[0]):
        cluster_index = belong_to[i]
        result_expected = cluster_to_digit[cluster_index]
        if (result_expected != Y_training[i]):
            wrong_predict += 1

    return wrong_predict * 1.0 / X_training.shape[0]

def k_mean(X_training, Y_training, itr_time, generate_centers, k):
    all_centers = generate_centers(X_training, k)
    all_centers = np.array(all_centers)

    pre_belong_to = [-1] * k
    belong_to = [-2] * k
    for t in range(0, itr_time):
        belong_to = find_close_centers(X_training, all_centers)

        if (belong_to == pre_belong_to):
            print("converge at iteration {0} for k = {1}".format(t + 1, k - 1))
            break
        else:
            pre_belong_to = belong_to

        all_centers = recompute_center(belong_to, X_training, k)
        # print(all_centers)

    sum_sqr = compute_sum_sqr(belong_to, all_centers, X_training)
    total_mistake_rate = compute_total_mistake_rate(belong_to, X_training, Y_training, k)

    return (sum_sqr, total_mistake_rate)

def execute(X_training, Y_training, itr_time, generate_centers_func):
    k_array = []
    sum_sqr_array = []
    mistake_rate_array = []

    for k in range(1, 11):
        (cur_sum_sqr, cur_mistake_rate) = k_mean(X_training, Y_training, 20, generate_centers_func, k)
        k_array.append(k)
        sum_sqr_array.append(cur_sum_sqr)
        mistake_rate_array.append(cur_mistake_rate)
        print("when k is {0}, sum square is {1}, mistake rate is {2}\n".format(k, cur_sum_sqr, cur_mistake_rate))

    # plt.plot(k_array, sum_sqr_array)
    plt.plot(k_array, mistake_rate_array)
    plt.show()

# 3.5.3, 3
execute(X_training, Y_training, 20, generate_start_centers_first_k)


# print(np.array([1,2,3]) * 1.0 / 2)
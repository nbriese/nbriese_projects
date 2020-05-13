# Kernel SVM
# Copyright Nathan Briese 2020

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvxopt

def split_data_percent(data, test_percent):
    test_percent /= 100.0
    num_test = int(len(data) * test_percent)
    num_train = len(data) - num_test
    train_data = data[:num_train]
    test_data = data[-num_test:]
    return train_data, test_data

def split_features_labels(data):
    # Splits the features from the labels
    data = np.transpose(data)
    X = np.transpose(data[:-1])
    y = np.transpose(data[-1])
    return X, y

def kernel(x, y, sigma):
    # Calculate the Gaussian kernel function
    return np.exp((-np.linalg.norm(x - y) ** 2) / (2 * sigma **2))

def rbf_svm_train(X, y, c, sigma):
    y_diag = np.diag(y)
    size = len(X)
    P = np.identity(len(X))
    for i in range(size):
        for j in range(len(X)):
            P[i][j] = kernel(X[i], X[j], sigma)
    P = y_diag.dot(P.dot(y_diag))

    q  = - np.ones((len(X), 1))
    h1 = np.zeros((len(X), 1))
    h2 = c * (np.ones((len(X), 1)))
    h  = np.concatenate((h1, h2))
    g1 = np.diag(-np.ones(len(X)))
    g2 = np.diag(np.ones(len(X)))
    G  = np.concatenate((g1, g2))

    P = cvxopt.matrix(P, tc='d')
    q = cvxopt.matrix(q, tc='d')
    G = cvxopt.matrix(G, tc = 'd')
    h = cvxopt.matrix(h, tc = 'd')

    cvxopt.solvers.options['show_progress'] = False
    sol = cvxopt.solvers.qp(P,q,G,h)
    lambdas = np.array(sol['x'])

    return lambdas

def rbf_svm_predict(test_feats, test_lbls, train_feats, train_lbls, alphas, sigma):
    # predicts the labels using the model
    # calculate accuracy for the given test data
    labels = np.zeros(len(test_feats))
    acc = 0
    for i in range(len(test_feats)):
        sum = 0
        for j in range(len(train_feats)):
            sum += train_lbls[j] * alphas[j] * kernel(train_feats[j], test_feats[i], sigma)
        if sum >= 0:
            labels[i] = 1
        else:
            labels[i] = -1
        if labels[i] == test_lbls[i]:
            acc += 1

    acc /= len(test_data)

    return acc

##############          DRIVER SECTION          ###############

raw_data = pd.read_csv("XOR_data.csv").to_numpy()

# use 20% of the data as the test set
train_data, test_data = split_data_percent(raw_data, 20)
train_feats, train_lbls = split_features_labels(train_data)
test_feats,  test_lbls  = split_features_labels(test_data)

# try different values for the misclassification weight
# C = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
C = [0.1]

# try different values for the varience
# sigma = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
sigma = [0.1]

# training accuracy is not taken into account because it will always be 1
test_acc = np.zeros((len(C), len(sigma)))

# reporting the accuracy as C and σ vary
for i in range(len(C)):
    for j in range(len(sigma)):
        print("Starting training for c =", C[i], "σ =", sigma[j])
        alphas = rbf_svm_train(train_feats, train_lbls, C[i], sigma[j])
        test_acc[i][j] = rbf_svm_predict(test_feats, test_lbls, train_feats, train_lbls, alphas, sigma[j])
        np.savetxt("kernel_svm_model_" + str(i) + "_" + str(j) + ".csv", alphas, delimiter=',')
        print("Test accuracy", test_acc[i][j])

# Since Gaussian kernel has two hyper-parameters heat map is used for the plot
# plt.imshow(test_acc)
# plt.ylabel("Varying C")
# plt.xlabel("Varying σ")
# plt.colorbar()
# plt.title("Test Accuracy with varying C and σ")
# plt.show()

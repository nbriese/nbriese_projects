# Linear Support Vector Machine
# Copyright Nathan Briese 2020

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvxopt

def split_data_percent(data, test_percent):
    # Split this dataset into % split and hold out the last % to be used as test dataset.
    test_percent /= 100
    num_test = int(len(data) * test_percent)
    num_train = len(data) - num_test
    train_data = data[:num_train]
    test_data = data[-num_test:]
    return train_data, test_data

def split_features_labels(data):
    # Splits the features from the labels
    data = np.transpose(data)
    return np.transpose(data[:-1]), np.transpose(data[-1])

def linear_svm_train(X, y, c):
    y_diag = np.diag(y)
    p_matrix = y_diag.dot(X.dot(X.T.dot(y_diag)))
    q_matrix = -np.ones((len(X),1))
    h1 = np.zeros((len(X),1))
    h2 = c*(np.ones((len(X),1)))
    h_matrix = np.concatenate((h1,h2))
    g1 = np.diag(-np.ones(len(X)))
    g2 = np.diag(np.ones(len(X)))
    G_matrix = np.concatenate((g1,g2))

    P = cvxopt.matrix(p_matrix, tc='d')
    q = cvxopt.matrix(q_matrix, tc='d')
    G = cvxopt.matrix(G_matrix, tc = 'd')
    h = cvxopt.matrix(h_matrix, tc = 'd')

    cvxopt.solvers.options['show_progress'] = False
    sol = cvxopt.solvers.qp(P,q,G,h)
    lambdas = np.array(sol['x'])
    weight = lambdas.T.dot(y_diag.dot(X))
    return weight.T

def predict(X, weight):
    # label = predict(X, weight) to predicts the labels using the model
    labels = np.zeros(len(X))
    for i in range(len(X)):
        if weight[0]*X[i][0] + weight[1]*X[i][1] >= 0:
            labels[i] = 1.0
        else:
            labels[i] = -1.0
    return labels

def get_accuracy(feats, lbls, weight):
    acc = 0
    labels = predict(feats, weight)
    for i in range(len(feats)):
        if labels[i] == lbls[i]:
            acc += 1
    acc /= len(feats)
    return acc

##############          DRIVER SECTION          ###############

raw_data = pd.read_csv("XOR_data.csv").to_numpy()

# use 20% of the data as the test set
train_data, test_data = split_data_percent(raw_data, 20)
train_feats, train_lbls = split_features_labels(train_data)
test_feats,  test_lbls  = split_features_labels(test_data)

# try different values for the misclassification weight
# C = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
C = [0.1, 1, 10]

train_acc = np.zeros(len(C))
test_acc = np.zeros(len(C))

# report the average train, validation and test accuracy as C varies.
for i in range(len(C)):
    print("Starting training for c=", C[i])
    weight = linear_svm_train(train_feats, train_lbls, C[i])
    np.savetxt("linear_svm_model_" + str(i) + ".csv", weight, delimiter=',')
    train_acc[i] = get_accuracy(train_feats, train_lbls, weight)
    test_acc[i]  = get_accuracy(test_feats, test_lbls, weight)

    print("Training accuracy", train_acc[i])
    print("Test     accuracy", test_acc[i])

# Generate a plot
C_names = [str(c) for c in C]
plt.plot(C_names, train_acc, label='Training')
plt.plot(C_names, test_acc, label='Test')
plt.ylabel("Accuracy")
plt.xlabel("Different Values of C")
plt.title("Accuracy of Linear SVM with Varying c")
plt.legend()
plt.show()

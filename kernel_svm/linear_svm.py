# Linear Support Vector Machine
# Copyright Nathan Briese 2020

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvxopt

def split_data_k_portions(data, k):
    # splits the data into k portions
    # last portion might be short by a few entries
    num_entries = int(len(data)/k) + 1
    start = (k-1)*num_entries
    return data[0:start], data[start:]

def split_features_labels(data):
    # Splits the features from the labels
    data = np.transpose(data)
    X = np.transpose(data[:-1])
    y = np.transpose(data[-1])
    return X, y
    # return data[:][:-1], data[-1]

def svmfit(X, y, c):
    # to train the model
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
    weight = weight.T
    return weight

def predict(X, weight):
    # label = predict(X, weight) to predicts the labels using the model
    # weight can be either ndarray or list
    labels = np.zeros(len(X))
    for i in range(len(X)):
        if weight[0]*X[i][0] + weight[1]*X[i][1] >= 0:
            labels[i] = 1.0
        else:
            labels[i] = -1.0
    return labels

def k_fold_cv(train_data, test_data, k, c):
    # train accuracy, cv accuracy, test accuracy = k fold cv(train data, test data, k, c)
    # k-fold cross validation
    train_accuracy = np.zeros(k)
    cv_accuracy = np.zeros(k)
    test_accuracy = np.zeros(k)
    for i in range(k):
        split_training_data, split_val_data = split_data_k_portions(train_data, k)

        # train on the data
        train_feats, train_lbls = split_features_labels(split_training_data)
        weights = svmfit(train_feats, train_lbls, c)

        #calculate training error
        train_l = predict(train_feats, weights)
        for j in range(len(train_feats)):
            if train_l[j] == train_lbls[j]:
                train_accuracy[i] += 1
        train_accuracy[i] /= len(train_feats)


        #calculate validation error
        val_feats, val_lbls = split_features_labels(split_val_data)
        val_l = predict(val_feats, weights)
        for j in range(len(val_feats)):
            if val_l[j] == val_lbls[j]:
                cv_accuracy[i] += 1
        cv_accuracy[i] /= len(val_feats)

        #calculate testing error
        test_feats, test_lbls = split_features_labels(test_data)
        test_l = predict(test_feats, weights)
        for j in range(len(test_feats)):
            if test_l[j] == test_lbls[j]:
                test_accuracy[i] += 1
        test_accuracy[i] /= len(test_feats)

    t_acc = np.mean(train_accuracy)
    v_acc = np.mean(cv_accuracy)
    tst_acc = np.mean(test_accuracy)
    return t_acc, v_acc, tst_acc


def split_data_percent(data, test_percent):
    # Split this dataset into % split and hold out the last % to be used as test dataset.
    test_percent /= 100
    num_test = int(len(data) * test_percent)
    num_train = len(data) - num_test
    train_data = data[:num_train]
    test_data = data[-num_test:]
    return train_data, test_data

##############          DRIVER SECTION          ###############

raw_data = pd.read_csv("XOR_data.csv").to_numpy()
train_data, test_data = split_data_percent(raw_data, 20)

k = 10
C = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]

train_acc = np.zeros(len(C))
cv_acc = np.zeros(len(C))
test_acc = np.zeros(len(C))

# report the average train, validation and test accuracy as C varies.
for i in range(len(C)):
    train_acc[i], cv_acc[i], test_acc[i] = k_fold_cv(train_data, test_data, k, C[i])
    print("train_accuaracy for c = ", C[i], " is ", train_acc[i])
    print("cv_accuaracy for c = ", C[i], " is ", cv_acc[i])
    print("test_accuaracy for c = ", C[i], " is ", test_acc[i])

print("Train: ", train_acc)
print("Val: ", cv_acc)
print("Test: ", test_acc)

# Generate the plots.
C_names = str(C)
plt.plot(C_names, train_acc, label='Training')
plt.plot(C_names, cv_acc, label='Validation')
plt.plot(C_names, test_acc, label='Test')
plt.ylabel("Accuracy %")
plt.xlabel("Different Values of c")
plt.title("Accuracy of Linear SVM with Varying c")
plt.legend()
plt.show()
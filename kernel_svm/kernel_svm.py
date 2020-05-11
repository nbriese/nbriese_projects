# Kernel SVM
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

def kernel(x, y, sigma):
    # Calculate the Gaussian kernel function
    return np.exp((-np.linalg.norm(x - y)**2)/(2*sigma**2))

def rbf_svm_train(X, y, c, sigma):  
    # α = rbf svm train(X, y, c, σ) to train your model
    y_diag = np.diag(y)
    size = len(X)
    P = np.identity(len(X))
    for i in range(size):
        for j in range(len(X)):
            P[i][j] = kernel(X[i], X[j], sigma)
    P = y_diag.dot(P.dot(y_diag))

    q = -np.ones((len(X),1))
    h1 = np.zeros((len(X),1))
    h2 = c*(np.ones((len(X),1)))
    h = np.concatenate((h1,h2))
    g1 = np.diag(-np.ones(len(X)))
    g2 = np.diag(np.ones(len(X)))
    G = np.concatenate((g1,g2))

    P = cvxopt.matrix(P, tc='d')
    q = cvxopt.matrix(q, tc='d')
    G = cvxopt.matrix(G, tc = 'd')
    h = cvxopt.matrix(h, tc = 'd')

    cvxopt.solvers.options['show_progress'] = False
    sol = cvxopt.solvers.qp(P,q,G,h)
    lambdas = np.array(sol['x'])

    return lambdas

def rbf_svm_predict(test_X, train_X, train_y, alpha, sigma):
    # label = predict(test X, train X, train y, α, σ) to predicts the labels using the model
    labels = np.zeros(len(test_X))
    for i in range(len(test_X)):
        sum = 0
        for j in range(len(train_X)):
            sum += train_y[j] * alpha[j] * kernel(train_X[j], test_X[i], sigma)
        if sum >= 0:
            labels[i] = 1
        else:
            labels[i] = -1
    return labels
    
def k_fold_cv(train_data, test_data, k, c, sigma):
    # train accuracy, cv accuracy, test accuracy = k fold cv(train data, test data, k, c)
    # k-fold cross validation
    cv_accuracy = np.zeros(k)
    test_accuracy = np.zeros(k)
    print("Starting", k, "fold cross validation with parameters", c, sigma)
    for i in range(k):
        print("Starting fold:", i)
        split_training_data, split_val_data = split_data_k_portions(train_data, k)

        # train on the data
        train_feats, train_lbls = split_features_labels(split_training_data)
        alphas = rbf_svm_train(train_feats, train_lbls, c, sigma)
        np.savetxt("kernel_svm_model.csv", alphas, delimiter=',')

        #calculate validation error
        val_feats, val_lbls = split_features_labels(split_val_data)
        val_l = rbf_svm_predict(val_feats, train_feats, train_lbls, alphas, sigma)
        for j in range(len(val_feats)):
            if val_l[j] != val_lbls[j]:
                cv_accuracy[i] += 1
        cv_accuracy[i] /= len(val_feats)
        print("Validation error rate for fold number", i, "is", cv_accuracy[i])

        #calculate testing error
        test_feats, test_lbls = split_features_labels(test_data)
        test_l = rbf_svm_predict(test_feats, train_feats, train_lbls, alphas, sigma)
        for j in range(len(test_feats)):
            if test_l[j] != test_lbls[j]:
                test_accuracy[i] += 1
        test_accuracy[i] /= len(test_feats)
        print("Test       error rate for fold number", i, "is", test_accuracy[i])

    v_acc = np.mean(cv_accuracy)
    tst_acc = np.mean(test_accuracy)
    return v_acc, tst_acc


##############          DRIVER SECTION          ###############

raw_data = pd.read_csv("XOR_data.csv").to_numpy()
train_data, test_data = split_data_percent(raw_data, 20)
 
# implement k=10 fold cross validation on the first 80% of the data as split above.
k = 10
# C = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
C = [0.1]
# sigma = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
sigma = [0.1]
cv_acc = np.zeros((len(C), len(sigma)))
test_acc = np.zeros((len(C), len(sigma)))

# Please report the error rates as C and σ vary
for i in range(len(C)):
    for j in range(len(sigma)):
        cv_acc[i][j], test_acc[i][j] = k_fold_cv(train_data, test_data, k, C[i], sigma[j])
        print("Validation error rate for c = ", C[i], " σ = ", sigma[j], " is ",  cv_acc[i][j])
        print("Test error rate for c = ", C[i], " σ = ", sigma[j], " is ", test_acc[i][j])
    
print("Validation error over C and σ:\n", cv_acc)
print("Test error over C and σ:\n", test_acc)

# Since RBF kernel has two hyper-parameters heat map is used for the plot
# plt.imshow(cv_acc)
# plt.ylabel("Varying C")
# plt.xlabel("Varying σ")
# plt.colorbar()
# plt.title("Validation Error with varying C and σ")
# plt.show()

# plt.imshow(test_acc)
# plt.ylabel("Varying C")
# plt.xlabel("Varying σ")
# plt.colorbar()
# plt.title("Test Error with varying C and σ")
# plt.show()

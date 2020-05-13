# Adaboost
# Copyright Nathan Briese 2020

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.datasets
from sklearn.tree import DecisionTreeClassifier

def train_learners(num_learners, features, labels):
    print("Staring training with", num_learners, "learners")
    learners = []
    alphas = []
    alpha_sum = 0
    dist = np.ones((num_learners,features.shape[0]))
    dist[0] /= features.shape[0]
    for i in range(num_learners):
        # print("Training learner: ", i)
        # create a weak learner based on the current distribution
        learners.append(
            DecisionTreeClassifier(max_depth=1)
            .fit(features, labels, sample_weight=dist[i])
        )
        
        # calculate epsilon
        epsilon = 0
        h_t = learners[i].predict(features)
        for j in range(len(labels)):
            if not (h_t[j] == labels[j]):
                epsilon += dist[i][j]
        if epsilon == 0:
            break
        
        # calculate alpha
        alpha = 0.5 * np.log((1-epsilon)/epsilon)
        alphas.append(alpha)
        alpha_sum += alpha

        # calculate new distribution
        # I avoided taking ln then exp to simmplify the calculation
        if(i < num_learners - 1):
            for j in range(features.shape[0]):
                if h_t[j] == labels[j]:
                    dist[i+1][j] = dist[i][j] * ((1-epsilon)/epsilon)**(-0.5)
                else:
                    dist[i+1][j] = dist[i][j] * ((1-epsilon)/epsilon)**(0.5)
            dist[i+1] /= np.sum(dist[i+1])

    return learners, alphas

def get_accuracy(learners, alphas, features, labels):
    accuracy = 0
    predictions = np.zeros((len(learners),len(features)))
    
    # pass the data through all learners
    for i in range(len(learners)):
        predictions[i] = learners[i].predict(features)

    # take a weighted vote from all learners
    for i in range(len(labels)):
        votes = np.zeros(max(labels) + 1)
        for j in range(len(learners)):
            votes[int(predictions[j][i])] += alphas[j]
        
        #update the accuracy
        if np.argmax(votes) == labels[i]:
            accuracy += 1

    accuracy /= len(labels)

    return accuracy

data = sklearn.datasets.load_breast_cancer()

train_feats = data.data[:500]
train_labels = data.target[:500]

test_feats = data.data[500:]
test_labels = data.target[500:]

del data

# train the ensemble
# ensemble, alphas = train_learners(40, train_feats, train_labels)

# see if it works
# print(get_accuracy(ensemble, alphas, train_feats, train_labels))
# print(get_accuracy(ensemble, alphas, test_feats, test_labels))

# Plot the misclassication error on both train and test set as the number of weak learners increase.
train_error_list = []
test_error_list = []
train_learner_nums = range(1, 200, 1)
test_learner_nums  = range(1, 200, 5)

for i in train_learner_nums:
    ensemble, alphas = train_learners(i, train_feats, train_labels)
    train_error_list.append(1 - get_accuracy(ensemble, alphas, train_feats, train_labels))

for i in test_learner_nums:
    ensemble, alphas = train_learners(i, train_feats, train_labels)
    test_error_list.append(1 - get_accuracy(ensemble, alphas, test_feats, test_labels))

fig = plt.figure()
plt.plot(train_learner_nums, train_error_list, label='Training Error')
plt.plot(test_learner_nums, test_error_list, label='Testing Error')
plt.legend()
plt.ylabel("Misclassification Error")
plt.xlabel("Number of Weak Learners")
plt.title("Training and Test Error Rate as the Number of Weak Learners Increases")
# plt.show()
fig.savefig('ada_num_learners.png')

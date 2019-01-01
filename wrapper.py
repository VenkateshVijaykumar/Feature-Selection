"""Authored by: Venkatesh Vijaykumar github: @VenkateshVijaykumar
   This implements Wrapping as a Feature Selection Method.
   The wrapper essentially evaluates combinations of features and returns the set that
   provides best accuracy for the learner used. The learner in this case is the
   MLPClassifier from sklearn.
   The 'wrapper' interacts with the learner that uses the reduced feature set in order to find a subset
   that results in the best accuracy.
   The dataset used here in the main section is the Iris dataset also from sklearn.
   Requires Numpy and sklearn to run.
"""
"""To run:
   import the wrapper_method from this wrapper.py file into your own file.
   pass the features and classes in array formats [n_samples,n_features] for feature array and
   [n_samples] or [n_samples,n_outputs] for class array.
   You can vary the percent_reduction parameter to reflect the percentage of original features
   to be retained. The retained percentage will include the most important n% of the original
   Can set the estimator to be any sklearn learner implementing the fit() and predict() methods
"""

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from matplotlib import pyplot as plt
from itertools import combinations

def wrapper_method(features,classes,estimator,percent_reduction=0.5,n_est=10):
    idx_lst = []
    combs = []
    acc_lst = []
    for i in range(features.shape[1]):
        idx_lst.append(i)
    for i in xrange(1, len(idx_lst)+1):
        els = [list(x) for x in combinations(idx_lst, i)]
        combs.extend(els)
    for i in range(len(combs)):
        new_features = features[:,combs[i]][:]
        estimator.fit(new_features,classes)
        predictions = estimator.predict(new_features)
        accuracy = accuracy_score(classes,predictions)*100
        acc_lst.append([combs[i],accuracy])
    best_comb, best_acc = max(acc_lst, key=lambda x:x[1])
    best_features = features[:,best_comb][:]
    return best_features, acc_lst
    
if __name__ == '__main__':
    iris = load_iris()
    train_feat = iris.data
    train_class = iris.target
    estimator = MLPClassifier(hidden_layer_sizes=100, max_iter=600, alpha=0.1, learning_rate='constant',
                    solver='adam', verbose=0, tol=1e-4, random_state=1, learning_rate_init=.05)
    best_combo, accuracies = et_filter_method(train_feat,train_class,estimator)
    print accuracies

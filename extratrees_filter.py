"""Authored by: Venkatesh Vijaykumar github: @VenkateshVijaykumar
   This implements Extra Trees as a Feature Selection Method.
   The tree method used is the ensemble Extra Trees method from sklearn.
   The et_filter_method acts as a feature 'filter' which performs feature selection based
   on the importance of the feature in the splits of the trees in the forest.
   The 'filter' has no interaction with the learner that uses the reduced feature set.
   The dataset used here in the main section is the Iris dataset also from sklearn.
   Requires Numpy and sklearn to run.
"""
"""To run:
   import the filter_method from this dt_filter.py file into your own file.
   pass the features and classes in array formats [n_samples,n_features] for feature array and
   [n_samples] or [n_samples,n_outputs] for class array.
   You can vary the percent_reduction parameter to reflect the percentage of original features
   to be retained. The retained percentage will include the most important n% of the original
   You can vary the number of weak learners in the ensemble with the n_est parameter
"""

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_iris
from matplotlib import pyplot as plt

def et_filter_method(features,classes,percent_reduction=0.5,n_est=10):
    clf = ExtraTreesClassifier(n_estimators = n_est, criterion = "gini", random_state = 100, max_depth=None, min_samples_leaf=5)
    clf.fit(features, classes)
    filtered_features = np.argsort(clf.feature_importances_).tolist()
    filtered_features = filtered_features[::-1]
    num_features = int(np.ceil(percent_reduction * len(filtered_features)))
    final_features = filtered_features[:num_features]
    new_features = features[:,final_features][:]
    return new_features, np.asarray(filtered_features)

if __name__ == '__main__':
    iris = load_iris()
    train_feat = iris.data
    train_class = iris.target
    print "Original feature space shape is: ", train_feat.shape
    new_train_feat, sorted_features = et_filter_method(train_feat,train_class)
    print "Features in order of importance are: ", sorted_features
    print "Reduced feature space shape is: ", new_train_feat.shape
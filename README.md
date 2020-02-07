# Feature-Selection

This repository contains Python programs for feature selection routines for ML algorithms, based on ML algorithms. Unlike feature transformation, these routines retain the most important *N percent* of the original feature set.
The selection routines were inspired by the excellent content in the CS-7641 class at Georgia Tech.


# Files

The files in this repo are:

 - decisiontree_filter.py:
 - extratrees_filter.py
 - randomforest_filter.py
 - wrapper.py

The decisiontree_filter, extratrees_filter and randomforest_filter utilize the Gini index to determine the importance of the feature. The filter methods then return the new truncated feature set that retain the user specified *N percent* most important features.

The wrapper routine considers all possible combinations of features, and calculates the accuracy of the estimator for each of these. It returns the combination with the best accuracy, and also a list of combinations with their associated accuracy values.

# Requirements

Packages required for the programs in this repository to run are:

 - NumPy
 - Scikit-Learn

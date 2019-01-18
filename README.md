# Feature-Selection
Code for basic feature selection algorithms, inspired by the content from CS7641 at Georgia Tech

This implements Decision Trees, Random Forests, and Extra Trees as Feature Selection Methods. The methods act as  feature 'filters' which perform feature selection based on the importance of the feature in course of the particular algorithm. The 'filter' has no interaction with the learner that uses the reduced feature set.
This also implements Wrapping as a Feature Selection Method. The wrapper essentially evaluates combinations of features and returns the set that provides best accuracy for the learner used. The learner in this case is the MLPClassifier from sklearn. The 'wrapper' interacts with the learner that uses the reduced feature set in order to find a subset that results in the best accuracy

The dataset used here in the main section is the Iris dataset also from sklearn.
Requires Numpy and sklearn to run.

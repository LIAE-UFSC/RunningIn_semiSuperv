from sklearn import semi_supervised
from sklearn import svm
from sklearn.neighbors import (NeighborhoodComponentsAnalysis, KNeighborsClassifier)
from sklearn.linear_model import SGDClassifier
from sklearn import tree
from sklearn.cross_decomposition import PLSCanonical
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

# TODO: Use this file to write wrapper functions for the classifiers and select the model based on the classifier name.

def select_model(classifier, **kwargs):
    # TODO: write docstring

    supervised_classifiers = None # Placeholder for supervised classifiers

    # Build the semi-supervised model
    semisupervised_model = semi_supervised.SelfTrainingClassifier(base_estimator=supervised_classifiers, threshold=0.8)

    return semisupervised_model


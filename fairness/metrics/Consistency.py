from fairness.metrics.Metric import Metric
from sklearn.metrics import pairwise_distances
import numpy as np
import pandas as pd

class Consistency(Metric):
    def __init__(self):
        Metric.__init__(self)
        self.name = 'indiv_fairness_consistency'
        self.n_neighbors = 5

    def euclidian_distance(self, a, b):
        ''' Euclidian Distance Function '''
        return np.sqrt(np.sum(np.square(a - b)))

    def calc(self, actual, predicted, dict_of_sensitive_lists, single_sensitive_name, 
        unprotected_vals, positive_pred, features):

        # Skip for categorical features
        categorical_columns =  features.select_dtypes(include = ['object']).columns.tolist()
        if len(categorical_columns) != 0:
            features = pd.get_dummies(features, columns = categorical_columns)

        predicted = np.asarray(predicted)

        distMat = pairwise_distances(features, metric = self.euclidian_distance)
        knn_ixs = np.argsort(distMat, axis = 1)[:, 1:] # sort neighbors
        knn_ixs = knn_ixs[:, 0 : self.n_neighbors] # only keep n_neighbors

        consistencies = ([np.abs(predicted[i] - 
            np.mean(predicted[knn_ixs[i]])) for i in range(len(predicted))])

        return (1.0 - np.mean(consistencies))

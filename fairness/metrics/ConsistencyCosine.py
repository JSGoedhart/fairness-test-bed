from fairness.metrics.Metric import Metric
from sklearn.metrics import pairwise_distances
import numpy as np
import pandas as pd

class ConsistencyCosine(Metric):
    def __init__(self):
        Metric.__init__(self)
        self.name = 'indiv_fairness_consistency_cosine'
        self.n_neighbors = 5

    def cosine_distance(self, a, b):
        ''' Cosine similarity Function '''
        return np.dot(a, b)/(np.linalg.norm(a) * np.linalg.norm(b))

    def calc(self, actual, predicted, dict_of_sensitive_lists, single_sensitive_name, 
        unprotected_vals, positive_pred, features):

        # Skip for categorical features
        categorical_columns =  features.select_dtypes(include = ['object']).columns.tolist()
        if len(categorical_columns) != 0:
            features = pd.get_dummies(features, columns = categorical_columns)

        # Transform predicted array and get length
        predicted = np.asarray(predicted)
        n = predicted.shape[0]

        # Compute distances
        distMat = pairwise_distances(features, metric = self.cosine_distance)

        # Select K+1 last elements from argsort (min to max) and leave out last element with (n - 1)
        knn_ixs = np.argsort(distMat, axis = 1)[:,-(self.n_neighbors+1):(n - 1)]

        # Compute consistencies
        consistencies = ([np.abs(predicted[i] - 
            np.mean(predicted[knn_ixs[i]])) for i in range(len(predicted))])

        return (1.0 - np.mean(consistencies))

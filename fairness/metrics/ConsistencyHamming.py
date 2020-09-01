from fairness.metrics.Metric import Metric
from sklearn.metrics import pairwise_distances
import numpy as np
import pandas as pd

class ConsistencyHamming(Metric):
    def __init__(self):
        Metric.__init__(self)
        self.name = 'indiv_fairness_consistency_hamming'
        self.n_neighbors = 5

    def calc(self, actual, predicted, dict_of_sensitive_lists, single_sensitive_name, 
        unprotected_vals, positive_pred, features):
        
        # Dummify categorical features
        categorical_columns =  features.select_dtypes(include = ['object']).columns.tolist()
        if len(categorical_columns) != 0:
            features = pd.get_dummies(features, columns = categorical_columns)

        # Remove non dummy features
        bool_cols = [col for col in features if np.isin(features[col].dropna().unique(), [0, 1]).all()]
        features = features[bool_cols]

        predicted = np.asarray(predicted)

        distMat = pairwise_distances(features, metric = 'hamming')

        knn_ixs = np.argsort(distMat, axis = 1)[:, 1:] # sort neighbors
        knn_ixs = knn_ixs[:, 0 : self.n_neighbors] # only keep n_neighbors

        consistencies = ([np.abs(predicted[i] - 
            np.mean(predicted[knn_ixs[i]])) for i in range(len(predicted))])

        return (1.0 - np.mean(consistencies))

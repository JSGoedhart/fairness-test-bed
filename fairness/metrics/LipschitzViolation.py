from fairness.metrics.Metric import Metric
from sklearn.metrics import pairwise_distances
import numpy as np
import pandas as pd
import os
import pickle

class LipschitzViolation(Metric):
    def __init__(self, K, distance_function = 'euclidean'):
        Metric.__init__(self)
        self.name = 'lipschitz_' + distance_function + '_' + str(K)
        self.distance_function = distance_function
        self.K = K

    def total_variation_distance(self, a, b):
        '''  Total variation distance as described in Dwork (2018) '''
        return np.abs(a[0] - b[0])

    def calc(self, actual, predicted, dict_of_sensitive_lists, single_sensitive_name, 
        unprotected_vals, positive_pred, indices, predicted_probs, dataset):
    
        # Load distance matrix over features
        distance_path = os.path.join(os.getcwd(), 'fairness', 'data', 'distance_matrices')
        file = os.path.join(distance_path, dataset + '_' + 'numerical-binsensitive_distance_matrix'
            + '_' + self.distance_function + '.pkl')
        matrix = pickle.load(open(file, 'rb'))
                
        # Get correct entries
        temp = matrix[indices, :]
        dist_mat_x = temp[:, indices]

        # Get distance matrix over prediction probabilities
        dist_mat_y = pairwise_distances(predicted_probs, metric = self.total_variation_distance)
        
        violations = (dist_mat_x < self.K * dist_mat_y).sum()
        
        return violations / (dist_mat_x.shape[0] * dist_mat_x.shape[0] - dist_mat_x.shape[0])

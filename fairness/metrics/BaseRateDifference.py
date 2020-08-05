from fairness.metrics.Metric import Metric
from fairness.metrics.utils import calc_pos_protected_percents

class BaseRateDifference(Metric):
    ''' 
    Population Statistics: compute base rate difference 
    '''
    def __init__(self):
        Metric.__init__(self)
        self.name = 'baseratedifference'


    def calc(self, actual, predicted, dict_of_sensitive_lists, single_sensitive_name, 
        unprotected_vals, positive_pred, features = None):

        sensitive = dict_of_sensitive_lists[single_sensitive_name]
        unprotected_pos_percent, protected_pos_percent= calc_pos_protected_percents(actual, 
            sensitive, unprotected_vals, positive_pred)
        baseratedifference = protected_pos_percent - unprotected_pos_percent

        return baseratedifference
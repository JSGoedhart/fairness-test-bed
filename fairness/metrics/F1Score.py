from fairness.metrics.Metric import Metric
from sklearn.metrics import f1_score

class F1Score(Metric):
    """
    Returns the true positive rate (aka recall) for the predictions.  Assumes binary
    classification.
    """
    def __init__(self):
        Metric.__init__(self)
        self.name = 'F1-Score'

    def calc(self, actual, predicted, dict_of_sensitive_lists, single_sensitive_name,
             unprotected_vals, positive_pred, features = None):
        return f1_score(actual, predicted, pos_label=positive_pred, average='binary')


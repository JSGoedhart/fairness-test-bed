from fairness.algorithms.Algorithm import Algorithm
from fairness.algorithms.zemel.lfr_cosine import LFR

class ZemelAlgorithm(Algorithm):

	def __init__(self):
		Algorithm.__init__(self)
		self.name = "Zemel"

	def run(self, train_df, test_df, class_attr, positive_class_val, sensitive_attrs, 
		single_sensitive, privileged_vals, params):

		''' First Compute the transformation parameters that leads to fair representations. '''

		X_train, X_test = train_df.loc[:, train_df.columns != class_attr], test_df.loc[:, test_df.columns != class_attr]
		y_train, y_test = train_df[class_attr], test_df[class_attr]
		sens_attr_name = single_sensitive
		priv_group_value = privileged_vals[sensitive_attrs.index(single_sensitive)]
		unpriv_group_value = 1 if priv_group_value == 0 else 0

		# Fit, learn model parameters based on train set
		model = LFR()
		model = model.fit(X_train, y_train, sens_attr_name, priv_group_value, unpriv_group_value)

		# Transform test set and predict
		transformed_labels, transformed_labels_binary = model.transform(X_test, y_test, sens_attr_name, priv_group_value, unpriv_group_value)
		
		return [int(i) for i in transformed_labels_binary], []

	def get_supported_data_types(self):
		return set(["numerical-binsensitive"])
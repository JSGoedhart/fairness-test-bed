import numpy as np
import scipy.optimize as optim
import fairness.algorithms.zemel.lfr_helpers_cosine as lfr_helpers

class LFR():
	""" Learning Fair Representations is a pre-processing technique that finds a
	latent representation which encodes the data well, but obfuscates information
	about protected attributes. """

	def __init__(self, 
		k = 5,
		Ax = 0.01, 
		Ay = 1.0,
		Az = 50.0,
		epsilon = 1e-5,
		print_interval = 100,
		seed = None,
		path_loss = None,
		maxfun = 10000,
		maxiter = 100):

		self.k = k
		self.Ax = Ax
		self.Ay = Ay
		self.Az = Az
		self.print_interval = print_interval
		self.seed = seed
		self.path_loss = path_loss
		self.learned_model = None
		self.maxfun = maxfun
		self.maxiter = maxiter
		self.epsilon = epsilon

	def fit(self, X_train, y_train, sens_attr_name, priv_group_value, unpriv_group_value):
		''' Compute the transformation parameters that leads to fair representations. '''

		if self.seed is not None:
			np.random.seed(self.seed)

		num_train_samples, features_dim = X_train.shape

		# Transform pandas data to numpy arrays
		sensitive_feature = X_train[sens_attr_name].values
		X_train = X_train.values
		y_train = y_train.values

		# Split into sensitive and nonsensitive groups
		sensitive_idx = np.array(np.where(sensitive_feature == unpriv_group_value)).flatten()
		nonsensitive_idx = np.array(np.where(sensitive_feature == priv_group_value)).flatten()
		training_sensitive = X_train[sensitive_idx]
		training_nonsensitive = X_train[nonsensitive_idx]
		ytrain_sensitive = y_train[sensitive_idx].reshape(-1, 1)
		ytrain_nonsensitive = y_train[nonsensitive_idx].reshape(-1, 1)

		# Specify initial parameter guess and parameter bounds
		model_inits = np.random.uniform(size = features_dim * 2 + self.k + features_dim * self.k)
		bnd = []
		for i, _ in enumerate(model_inits):
			# if i < features_dim * 2 or i >= features_dim * 2 + self.k:
			# 	bnd.append((None, None))
			# else:
			# 	bnd.append((None, None))
			# Joosje: adjust bounds, s.t. w > 0 and alphas > 0
			if i < 2 * features_dim:
				bnd.append((0, None)) # Alpha's should be non-negative and between 0 and 1 for cosine similarity for prototype/input
			elif i >= 2 * features_dim and i < 2 * features_dim + self.k:
				bnd.append((0, 1)) # Weights are boundes to be between 0 and 1
			else:
				bnd.append((0, 1)) # Prototypes should be positive (if input is positive only)

		fmin_l_bfgs = optim.fmin_l_bfgs_b(lfr_helpers.LFR_optim_obj, x0 = model_inits, epsilon = self.epsilon,
			args = (training_sensitive, training_nonsensitive, 
				ytrain_sensitive[:, 0], ytrain_nonsensitive[:, 0], 
				self.k, self.Ax, self.Ay, self.Az, False, self.print_interval, self.path_loss),
			bounds = bnd, approx_grad = True, maxfun = self.maxfun, maxiter = self.maxiter)

		self.learned_model = fmin_l_bfgs[0]
		self.function_value_min = fmin_l_bfgs[1]
		self.convergence_dict = fmin_l_bfgs[2]

		_, P = training_sensitive.shape
		alpha0 = self.learned_model[:P]
		alpha1 = self.learned_model[P : 2 * P]
		w = self.learned_model[2 * P : (2 * P) + self.k]
		v = np.matrix(self.learned_model[(2 * P) + self.k:]).reshape((self.k, P))

		return self

	def transform(self, X_test, y_test, sens_attr_name, priv_group_value, unpriv_group_value, threshold = 0.5):
		''' Transform the dataset using the learned model parameters '''

		if self.seed is not None:
			np.random.seed(self.seed)

		num_test_samples, _ = X_test.shape
		
		# Transform pandas data to numpy arrays
		sensitive_feature = X_test[sens_attr_name].values
		X_test = X_test.values
		y_test = y_test.values

		# Split into sensitive and nonsensitive groups
		sensitive_idx = np.array(np.where(sensitive_feature == unpriv_group_value)).flatten()
		nonsensitive_idx = np.array(np.where(sensitive_feature == priv_group_value)).flatten()
		testing_sensitive = X_test[sensitive_idx]
		testing_nonsensitive = X_test[nonsensitive_idx]
		ytest_sensitive = y_test[sensitive_idx].reshape(-1, 1)
		ytest_nonsensitive = y_test[nonsensitive_idx].reshape(-1, 1)

		# Extract parameters of trained model
		Ns, P = testing_sensitive.shape
		N, _ = testing_nonsensitive.shape
		alphaoptim0 = self.learned_model[:P] # Feature importances for nonsensitive group
		alphaoptim1 = self.learned_model[P: 2 * P] # Feature importances for sensitive group
		woptim = self.learned_model[2 * P: (2 * P) + self.k] # weight parameter per prototype, govern mapping to y_pred
		voptim = np.matrix(self.learned_model[(2 * P) + self.k:]).reshape((self.k, P)) # Matrix that maps input to k prototype vectors

		# Compute norm for cosine distances prototype / input
		X_norm_sensitive = lfr_helpers.compute_norm(testing_sensitive, alphaoptim1, Ns, P)
		X_norm_nonsensitive = lfr_helpers.compute_norm(testing_nonsensitive, alphaoptim0, N, P)

		# Compute distances between prototypes and original input
		dist_sensitive = lfr_helpers.distances_cosine(testing_sensitive, X_norm_sensitive, voptim, alphaoptim1, Ns, P, self.k)
		dist_nonsensitive = lfr_helpers.distances_cosine(testing_nonsensitive, X_norm_nonsensitive, voptim, alphaoptim0, N, P, self.k)

		# Use distances to compute cluster probabilities for test instances 
		M_nk_sensitive = lfr_helpers.M_nk_cosine(dist_sensitive, Ns, self.k)
		M_nk_nonsensitive = lfr_helpers.M_nk_cosine(dist_nonsensitive, N, self.k)

		# Compute learned mapping for test instances x_n_hat
		res_sensitive = lfr_helpers.x_n_hat_cosine(testing_sensitive, M_nk_sensitive, voptim, Ns, P, self.k)
		x_n_hat_sensitive = res_sensitive[0]
		res_nonsensitive = lfr_helpers.x_n_hat_cosine(testing_nonsensitive, M_nk_nonsensitive, voptim, N, P, self.k)
		x_n_hat_nonsensitive = res_nonsensitive[0]

		# print('shape x_n_hat_sensitive: ', x_n_hat_sensitive.shape)

		# Compute predictions for test instances
		res_sensitive = lfr_helpers.yhat(M_nk_sensitive, ytest_sensitive, woptim, Ns, self.k)
		y_hat_sensitive = res_sensitive[0]
		res_nonsensitive = lfr_helpers.yhat(M_nk_nonsensitive, ytest_nonsensitive, woptim, N, self.k)
		y_hat_nonsensitive = res_nonsensitive[0]

		# Transform features and labels
		transformed_features = np.zeros(shape=((Ns + N), P))
		transformed_labels = np.zeros(shape=np.shape(y_test))
		transformed_features[sensitive_idx] = x_n_hat_sensitive
		transformed_features[nonsensitive_idx] = x_n_hat_nonsensitive
		transformed_labels[sensitive_idx] = y_hat_sensitive
		transformed_labels[nonsensitive_idx] = y_hat_nonsensitive
		transformed_labels_binary = (np.array(transformed_labels) > threshold).astype(np.float64)

		return transformed_labels_binary, transformed_labels

	# def plot_alphas(self, P, column_names, fig_path = None):

	# 	# Plot feature importances for privileged and unprivileged group
	# 	alpha_min = self.learned_model[:P]
	# 	alpha_plus = self.learned_model[P : 2 * P]

	# 	fig, axs = plt.subplots(2)
	# 	fig.suptitle("Feature importances for LFR")
	# 	axs[0].bar(range(P), alpha_min, 
	# 		color="r", align="center")
	# 	axs[0].title.set_text('Alpha Minus (Unprivileged Group)')
	# 	axs[1].bar(range(P), alpha_plus, 
	# 		color="r", align="center")
	# 	axs[1].title.set_text('Alpha Plus (Privileged Group)')
	# 	plt.xticks(range(P), column_names, rotation = 'vertical')
	# 	plt.savefig(fig_path)
	# 	plt.close()



import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

current_path = os.getcwd()
results_path = os.path.join(current_path, 'fairness', 'results')

baselines = ['SVM', 'GaussianNB', 'LR', 'DecisionTree']

datasets = ['ricci_Race_numerical-binsensitive']

for dataset in datasets:

	results = pd.read_csv(os.path.join(results_path, dataset + '.csv'), sep = ',', usecols = ['algorithm', 'CV', 'baseratedifference', 'indiv_fairness_consistency', 'accuracy'])

	# Drop NAs
	results = results.dropna(axis = 0, how = 'any')

	# Select dataframe with fairness algorithms only
	results_fairness = results[~results["algorithm"].isin(baselines)]

	print(results_fairness)

	results_fairness['CV'] = results_fairness['CV'] - 1.0

	# Statistical Parity versus Base Rate Difference
	ax = sns.scatterplot(x = 'baseratedifference', y = 'CV', hue = 'algorithm', data = results_fairness)
	ax.set_title('Group Fairness versus Base Rate Difference for different algorithms that optimize for statistical parity')
	ax.set_ylabel('Statistical Parity: P[YPred=1|Unprivileged] - P[YPred=1|Privileged]')
	ax.set_xlabel('Base Rate Difference: P[Y=1|Unprivileged] - P[Y=1|Privileged]')
	plt.show()

	# Consistency versus Base Rate Difference
	ax = sns.scatterplot(x = 'baseratedifference', y = 'indiv_fairness_consistency', hue = 'algorithm', data = results_fairness)
	ax.set_title('Concistency (Individual Fairness) versus Base Rate Difference for different algorithms that optimize for statistical parity')
	ax.set_ylabel('Consistency (Zemel et al. 2013)')
	ax.set_xlabel('Base Rate Difference: P[Y=1|Unprivileged] - P[Y=1|Privileged]')
	plt.show()

	# Accuracy versus Base Rate Difference
	ax = sns.scatterplot(x = 'baseratedifference', y = 'accuracy', hue = 'algorithm', data = results_fairness)
	ax.set_title('Accuracy versus Base Rate Difference for different algorithms that optimize for statistical parity')
	ax.set_ylabel('Accuracy')
	ax.set_xlabel('Base Rate Difference: P[Y=1|Unprivileged] - P[Y=1|Privileged]')
	plt.show()

	# Statistical Parity versus Concistency
	ax = sns.scatterplot(x = 'indiv_fairness_consistency', y = 'CV', hue = 'algorithm', data = results_fairness)
	ax.set_title('Statistical Parity versus Concistency (Individual Fairness) for different algorithms that optimize for statistical parity')
	ax.set_ylabel('Statistical Parity: P[YPred=1|Unprivileged] - P[YPred=1|Privileged]')
	ax.set_xlabel('Consistency (Zemel et al. 2013)')
	plt.show()

	# Statistical Parity versus Accuracy
	ax = sns.scatterplot(x = 'accuracy', y = 'CV', hue = 'algorithm', data = results_fairness)
	ax.set_title('Statistical Parity versus Accuracy for different algorithms that optimize for statistical parity')
	ax.set_ylabel('Statistical Parity: P[YPred=1|Unprivileged] - P[YPred=1|Privileged]')
	ax.set_xlabel('Accuracy')
	plt.show()

	# Consistency versus Accuracy
	ax = sns.scatterplot(x = 'accuracy', y = 'indiv_fairness_consistency', hue = 'algorithm', data = results_fairness)
	ax.set_title('Concistency (Individual Fairness) versus Accuracy for different algorithms that optimize for statistical parity')
	ax.set_ylabel('Consistency (Zemel et al. 2013)')
	ax.set_xlabel('Accuracy')
	plt.show()
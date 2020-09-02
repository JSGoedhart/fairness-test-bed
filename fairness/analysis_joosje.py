import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.collections import PatchCollection
from itertools import combinations

current_path = os.getcwd()
results_path = os.path.join(current_path, 'fairness', 'results')
preprocessed_path = os.path.join(current_path, 'fairness', 'data', 'preprocessed')


baselines = ['SVM', 'GaussianNB', 'LR', 'DecisionTree']
datasets = ['ricci_Race_numerical-binsensitive', 'german_sex_numerical-binsensitive', 'german_age_numerical-binsensitive'] #['propublica-recidivism_race_numerical-binsensitive', 'propublica-recidivism_sex_numerical-binsensitive', 'german_sex_numerical-binsensitive', 'german_age_numerical-binsensitive', 'ricci_Race_numerical-binsensitive']
metrics = (['CV', 'indiv_fairness_consistency', 'accuracy', 
	'indiv_fairness_consistency_cosine'])#, 'indiv_fairness_consistency_hamming'])

metric_combinations = list(combinations(metrics, 2))

metric_maps = {
	'CV': 'Statistical Parity (Group Fairness)',
	'indiv_fairness_consistency': 'Consistency Euclidian (Individual Fairness)',
	'indiv_fairness_consistency_cosine': 'Consistency Cosine (Individual Fairness)',
	#'indiv_fairness_consistency_hamming': 'Consistency Hamming (Individual Fairness)',
	'accuracy': 'Accuracy'
}


colors = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf")

for dataset in datasets:

	results = pd.read_csv(os.path.join(results_path, dataset + '.csv'), sep = ',', usecols = ['algorithm'] + metrics)

	# Drop NAs
	results = results.dropna(axis = 0, how = 'any')

	# Select dataframe with fairness algorithms only
	results_fairness = results[~results["algorithm"].isin(baselines)]

	results_fairness['CV'] = results_fairness['CV'] - 1.0

	# Only select algorithms with good Demographic Parity
	rows_before = results_fairness.shape[0]
	results_fairness = results_fairness.loc[(results_fairness['CV'] >= - 0.10) & (results_fairness['CV'] <= 0.10)]

	print('Percentage of algorithms within Demographic Parity interval: ', results_fairness.shape[0] / rows_before)

	metrics_mean = results_fairness.groupby('algorithm').mean()
	metrics_std = results_fairness.groupby('algorithm').std()

	print(metrics_mean)
	print(metrics_std)

	algorithms = list(results_fairness['algorithm'].unique())
	nr_algorithms = len(algorithms)
	legend_objects = [Patch(facecolor = color, alpha = 0.5, edgecolor = color, label = algorithm) for color, algorithm in zip(colors[:nr_algorithms], algorithms)]

	for comb in metric_combinations:
		x, y = comb[0], comb[1]

		mean_xs, mean_ys = metrics_mean[x].tolist(), metrics_mean[y].tolist()
		std_xs, std_ys = metrics_std[x].tolist(), metrics_std[y].tolist()

		# Set borders
		x_min = min([mu_x - std_x for mu_x, std_x in zip(mean_xs, std_xs)])
		x_max = max([mu_x + std_x for mu_x, std_x in zip(mean_xs, std_xs)])
		y_min = min([mu_y - std_y for mu_y, std_y in zip(mean_ys, std_ys)])
		y_max = max([mu_y + std_y for mu_y, std_y in zip(mean_ys, std_ys)])

		fig, ax = plt.subplots(1)
		boxes = [Rectangle((mean_x - 0.5 * std_x, mean_y - 0.5 * std_y), std_x, std_y) for mean_x, mean_y, std_x, std_y in zip(mean_xs, mean_ys, std_xs, std_ys)]
		pc = PatchCollection(boxes, facecolor = colors, alpha = 0.5, edgecolor = colors, linewidths = 2)
		ax.add_collection(pc)
		plt.xlabel(metric_maps[x])
		plt.ylabel(metric_maps[y])
		plt.xlim(x_min, x_max)
		plt.ylim(y_min, y_max)
		plt.title(dataset)
		ax.legend(handles = legend_objects)
		plt.savefig(os.path.join(results_path, 'figs_joosje', 'sensitivity', dataset + '_' + x + '_' + y))

	# for combination in metric_combinations:

	# 	statistics = pd.DataFrame(columns = ['algorithm', combination[0], combination[1]])

	# 	fig, ax = plt.subplots(1)
	# 	x = 
	# 	boxes = [Rectangle((np.mean(x) - np.std(x), np.mean(y) - np.std(y)), 2 * np.std(x), 2 * np.std(y)) for x, y in zip(xdata, ydata)]
	# 	pc = PatchCollection(boxes, facecolor = ["#9b59b6", "#3498db"], edgecolor = ["#9b59b6", "#3498db"], linewidths=(2,), alpha = 0.5)
	# 	ax.add_collection(pc)
	# 	plt.show()

	# # Statistical Parity versus Base Rate Difference
	# ax = sns.scatterplot(x = 'baseratedifference', y = 'CV', hue = 'algorithm', data = results_fairness)
	# ax.set_title('Group Fairness vs Base Rate Difference for algorithms that optimize for statistical parity', fontsize = 10)
	# ax.set_ylabel('Statistical Parity: P[YPred=1|Unprivileged] - P[YPred=1|Privileged]', fontsize = 10)
	# ax.set_xlabel('Base Rate Difference: P[Y=1|Unprivileged] - P[Y=1|Privileged]', fontsize = 10)
	# plt.suptitle('Dataset: ' + dataset)
	# plt.savefig(os.path.join(results_path, 'figs_joosje', dataset + '_1'))
	# plt.close()

	# # Consistency versus Base Rate Difference
	# ax = sns.scatterplot(x = 'baseratedifference', y = 'indiv_fairness_consistency', hue = 'algorithm', data = results_fairness)
	# ax.set_title('Concistency (Individual Fairness) versus Base Rate Difference for algorithms that optimize for statistical parity', fontsize = 10)
	# ax.set_ylabel('Consistency (Zemel et al. 2013)', fontsize = 10)
	# ax.set_xlabel('Base Rate Difference: P[Y=1|Unprivileged] - P[Y=1|Privileged]', fontsize = 10)
	# plt.suptitle('Dataset: ' + dataset)
	# plt.savefig(os.path.join(results_path, 'figs_joosje', dataset + '_2'))
	# plt.close()

	# # Accuracy versus Base Rate Difference
	# ax = sns.scatterplot(x = 'baseratedifference', y = 'accuracy', hue = 'algorithm', data = results_fairness)
	# ax.set_title('Accuracy versus Base Rate Difference for algorithms that optimize for statistical parity', fontsize = 10)
	# ax.set_ylabel('Accuracy', fontsize = 10)
	# ax.set_xlabel('Base Rate Difference: P[Y=1|Unprivileged] - P[Y=1|Privileged]', fontsize = 10)
	# plt.suptitle('Dataset: ' + dataset)
	# plt.savefig(os.path.join(results_path, 'figs_joosje', dataset + '_3'))
	# plt.close()

	# # Statistical Parity versus Concistency
	# ax = sns.scatterplot(x = 'CV', y = 'indiv_fairness_consistency', hue = 'algorithm', data = results_fairness)
	# ax.set_title('Statistical Parity versus Concistency (Individual Fairness) for algorithms that optimize for statistical parity', fontsize = 10)
	# ax.set_ylabel('Consistency (Zemel et al. 2013)', fontsize = 10)
	# ax.set_xlabel('Statistical Parity: P[YPred=1|Unprivileged] - P[YPred=1|Privileged]', fontsize = 10)
	# plt.suptitle('Dataset: ' + dataset)
	# plt.savefig(os.path.join(results_path, 'figs_joosje',  dataset + '_4'))
	# plt.close()

	# # Statistical Parity versus Accuracy
	# ax = sns.scatterplot(x = 'accuracy', y = 'CV', hue = 'algorithm', data = results_fairness)
	# ax.set_title('Statistical Parity versus Accuracy for algorithms that optimize for statistical parity', fontsize = 10)
	# ax.set_ylabel('Statistical Parity: P[YPred=1|Unprivileged] - P[YPred=1|Privileged]', fontsize = 10)
	# ax.set_xlabel('Accuracy', fontsize = 10)
	# plt.suptitle('Dataset: ' + dataset)
	# plt.savefig(os.path.join(results_path, 'figs_joosje',  dataset + '_5'))
	# plt.close()

	# # Consistency versus Accuracy
	# ax = sns.scatterplot(x = 'accuracy', y = 'indiv_fairness_consistency', hue = 'algorithm', data = results_fairness)
	# ax.set_title('Concistency (Individual Fairness) versus Accuracy for algorithms that optimize for statistical parity', fontsize = 10)
	# ax.set_ylabel('Consistency (Zemel et al. 2013)', fontsize = 10)
	# ax.set_xlabel('Accuracy', fontsize = 10)
	# plt.suptitle('Dataset: ' + dataset)
	# plt.savefig(os.path.join(results_path, 'figs_joosje',  dataset + '_6'))
	# plt.close()
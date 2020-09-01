import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from fairness.data.objects.list import DATASETS, get_dataset_names

current_path = os.getcwd()
preprocessed_path = os.path.join(current_path, 'fairness', 'data', 'preprocessed')
results_path = os.path.join(current_path, 'fairness', 'results', 'figs_joosje', 'data_distributions')


for dataset_obj in DATASETS:
	for sens_attr in dataset_obj.get_sensitive_attributes():
		dataset = dataset_obj.get_dataset_name() 
		target = dataset_obj.get_class_attribute()
		df = pd.read_csv(os.path.join(preprocessed_path, dataset + '_numerical-binsensitive.csv'), sep = ',', usecols = [target, sens_attr])

		if dataset == 'german':
			df[target] = df[target].replace({1.0: 1.0, 2.0: 0.0})

		sns.catplot(x = sens_attr, y = target, kind = 'bar', data = df)
		plt.xlabel(sens_attr + ' 0 is unprivileged')
		plt.ylabel(target)
		plt.title('Conditional target distribution for ' + dataset + ' and ' + sens_attr, fontsize = 10)
		plt.savefig(os.path.join(results_path, dataset + '_' + sens_attr))
		plt.close()
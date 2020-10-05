import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from fairness.data.objects.list import DATASETS, get_dataset_names


sns.set_context(rc={"figure.figsize": (8, 4)})
sns.set(style = 'white', font_scale = 1.5)


colors = ['#2ECC71' , '#FFEB3B'] # Green, yellow

current_path = os.getcwd()
preprocessed_path = os.path.join(current_path, 'fairness', 'data', 'preprocessed')
results_path = os.path.join(current_path, 'fairness', 'results', 'figs_joosje', 'data_distributions')


for dataset_obj in DATASETS:
	for sens_attr in dataset_obj.get_sensitive_attributes():
		dataset = dataset_obj.get_dataset_name() 
		target = dataset_obj.get_class_attribute()
		df = pd.read_csv(os.path.join(preprocessed_path, dataset + '_numerical-binsensitive.csv'), sep = ',', usecols = [target, sens_attr])

		print(dataset)
		print(df.shape)

		if dataset == 'german':
			df[target] = df[target].replace({1.0: 1.0, 2.0: 0.0})

		if 'propublica' in dataset:
			if 'violent' not in dataset:
				df[target] = df[target].replace({1.0:0.0, 0.0:1.0})

		sns.catplot(x = sens_attr, y = target, kind = 'bar', data = df, palette = colors, ci = None, edgecolor = 'black')
		sns.despine(top = False, right = False, left = False, bottom = False)
		plt.xlabel('A')
		plt.ylabel('Y')
		plt.ylim(0, 1.0)
		plt.savefig(os.path.join(results_path, dataset + '_' + sens_attr))
		plt.close()
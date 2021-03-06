import warnings
warnings.filterwarnings("ignore")


from fairness.data.objects.list import DATASETS, get_dataset_names
from fairness.data.objects.ProcessedData import ProcessedData


def main():
	dataset = get_dataset_names()

	for dataset_obj in DATASETS:
		if not dataset_obj.get_dataset_name() in dataset:
			continue
		if 'propublica' in dataset_obj.get_dataset_name():
			if 'violent' not in dataset_obj.get_dataset_name():
				print("\n Computing distances for dataset: " + dataset_obj.get_dataset_name())

				processed_dataset = ProcessedData(dataset_obj)

				# Compute distances
				processed_dataset.generate_distance_matrix(distance_metric = 'seuclidean')

if __name__ == '__main__':
	main()
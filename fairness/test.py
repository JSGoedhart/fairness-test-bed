from fairness.benchmark import run
import warnings
warnings.filterwarnings("ignore")


from fairness.data.objects.list import DATASETS, get_dataset_names
from fairness.data.objects.ProcessedData import ProcessedData


def main():

	run()


if __name__ == '__main__':
	main()
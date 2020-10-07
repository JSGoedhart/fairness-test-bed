#!/bin/bash

#SBATCH --job-name=benchmark_propublica
#SBATCH --time=01:00:00

srun python fairness/compute_distances.py
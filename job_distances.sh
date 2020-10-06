#!/bin/bash

#SBATCH --job-name=benchmark_propublica
#SBATCH --time=20:00:00

srun python fairness/compute_distances.py
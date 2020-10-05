#!/bin/bash

#SBATCH --job-name=compute_distances
#SBATCH --time=20:00:00

srun python fairness/compute_distances.py
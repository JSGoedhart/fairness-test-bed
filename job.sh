#!/bin/bash

#SBATCH --job-name=distances
#SBATCH --time=10:00:00

srun python fairness/benchmark.py
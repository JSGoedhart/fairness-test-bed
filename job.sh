#!/bin/bash

#SBATCH --job-name=distances
#SBATCH --time=100:00:00

srun python fairness/benchmark.py
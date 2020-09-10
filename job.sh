#!/bin/bash

#SBATCH --job-name=distances
#SBATCH --time=00:30:00

srun python fairness/benchmark.py
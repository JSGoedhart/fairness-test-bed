#!/bin/bash

#SBATCH --job-name=distances
#SBATCH --time=24:00:00

srun python fairness/benchmark.py
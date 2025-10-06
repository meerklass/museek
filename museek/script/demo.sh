#!/bin/bash

#SBATCH --job-name='MuSEEK-demo'
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=4GB
#SBATCH --output=museek-stdout.log
#SBATCH --error=museek-stderr.log
#SBATCH --time=00:05:00

source /path/to/virtualenv/museek/bin/activate
echo "Using a Python virtual environment: $(which python)"

echo "Submitting Slurm job"
museek museek.config.demo
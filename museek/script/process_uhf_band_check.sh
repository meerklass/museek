#!/bin/bash

#SBATCH --job-name='MuSEEK'
#SBATCH --cpus-per-task=26
#SBATCH --ntasks=1
#SBATCH --mem=128GB
#SBATCH --output=museek-stdout_task.log
#SBATCH --error=museek-stderr_task.log
#SBATCH --time=20:00:00

echo "Submitting Slurm job"

/users/wkhu/environment/museek/bin/python /users/wkhu/museek/cli/main.py museek.config.process_uhf_band


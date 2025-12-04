#!/bin/bash

#SBATCH --job-name='MuSEEK'
#SBATCH --cpus-per-task=26
#SBATCH --ntasks=1
#SBATCH --mem=192GB
#SBATCH --output=museek-stdout_task.log
#SBATCH --error=museek-stderr_task.log
#SBATCH --time=20:00:00

# Define repository directory as a variable
export MUSEEK_REPO_DIR="/users/yourname/museek"

# Log repository information
echo "Submitting Slurm job"
echo "Repository directory: $MUSEEK_REPO_DIR"

/users/yourname/environment/museek/bin/python $MUSEEK_REPO_DIR/cli/main.py museek.config.process_uhf_band
#!/bin/bash
#
#SBATCH --job-name=parameter_analysis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --mem=8G
#SBATCH --requeue

$RECOMMENDER_EXPERIMENT_PYTHON_BIN __main__.py configs/global.json \
-n 0 -s $SLURM_ARRAY_TASK_ID

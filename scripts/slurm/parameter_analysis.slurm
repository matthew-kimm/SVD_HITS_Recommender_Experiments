#!/bin/bash
#
#SBATCH --job-name=parameter_analysis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=5:00:00
#SBATCH --mem=8G
#SBATCH --requeue

$RECOMMENDER_EXPERIMENT_PYTHON_BIN __main__.py configs/parameter_analysis.json \
-n $RECOMMENDER_EXPERIMENT_PARAMETER_ANALYSIS_MIN_NEIGHBORS -s $SLURM_ARRAY_TASK_ID

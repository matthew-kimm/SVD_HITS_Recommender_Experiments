#!/bin/bash
#
#SBATCH --job-name=neighbor_experiments
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --mem=8G
#SBATCH --requeue

neighbor_start=`echo $RECOMMENDER_EXPERIMENT_NEIGHBOR_VARIATIONS | cut -d "," -f 1`
neighbor_step=`echo $RECOMMENDER_EXPERIMENT_NEIGHBOR_VARIATIONS | cut -d "," -f 2`
neighbor_end=`echo $RECOMMENDER_EXPERIMENT_NEIGHBOR_VARIATIONS | cut -d "," -f 3`

for f in ./configs/neighbor/*.json;
do
  for i in $(seq $neighbor_start $neighbor_step $neighbor_end);
  do
    $RECOMMENDER_EXPERIMENT_PYTHON_BIN __main__.py $f -n $i -s $SLURM_ARRAY_TASK_ID
  done
done

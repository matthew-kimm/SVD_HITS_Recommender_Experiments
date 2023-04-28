#!/bin/bash

jobid=`sbatch --array=1-$RECOMMENDER_EXPERIMENT_REPEATS --parsable ./scripts/slurm/neighbor_experiments.slurm | cut -d ":" -f 1`
sbatch --dependency=afterok:$jobid --wrap "touch make-targets/neighbor_experiments.completed"

counter=0
while ! [ -f make-targets/neighbor_experiments.completed ];
do
  sleep 60s
  counter=$(($counter + 1))
  echo "Waiting for Neighbor Experiments ... $counter minutes"
done
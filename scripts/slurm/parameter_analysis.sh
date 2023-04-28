#!/bin/bash

jobid=`sbatch --array=1-$RECOMMENDER_EXPERIMENT_REPEATS --parsable ./scripts/slurm/parameter_analysis.slurm | cut -d ":" -f 1`
sbatch --dependency=afterok:$jobid --wrap "touch make-targets/parameter_analysis.completed"

counter=0
while ! [ -f make-targets/parameter_analysis.completed ];
do
  sleep 60s
  counter=$(($counter + 1))
  echo "Waiting for Parameter Analysis ... $counter minutes"
done

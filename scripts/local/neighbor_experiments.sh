#!/bin/bash

neighbor_start=`echo $RECOMMENDER_EXPERIMENT_NEIGHBOR_VARIATIONS | cut -d "," -f 1`
neighbor_step=`echo $RECOMMENDER_EXPERIMENT_NEIGHBOR_VARIATIONS | cut -d "," -f 2`
neighbor_end=`echo $RECOMMENDER_EXPERIMENT_NEIGHBOR_VARIATIONS | cut -d "," -f 3`

for f in ./configs/neighbor/*.json;
do
  for i in $(seq $neighbor_start $neighbor_step $neighbor_end);
  do
    for j in $(seq $RECOMMENDER_EXPERIMENT_REPEATS);
    do
      $RECOMMENDER_EXPERIMENT_PYTHON_BIN __main__.py $f -n $i -s $j
    done
  done
done

touch make-targets/neighbor_experiments.completed

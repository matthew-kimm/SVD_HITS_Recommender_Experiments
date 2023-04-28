#!/bin/bash

for i in $(seq $RECOMMENDER_EXPERIMENT_REPEATS);
do
  echo $i
  $RECOMMENDER_EXPERIMENT_PYTHON_BIN __main__.py configs/parameter_analysis.json -n $RECOMMENDER_EXPERIMENT_PARAMETER_ANALYSIS_MIN_NEIGHBORS -s $i
done

touch make-targets/parameter_analysis.completed

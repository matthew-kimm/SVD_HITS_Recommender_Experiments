#!/bin/bash

for i in $(seq $RECOMMENDER_EXPERIMENT_REPEATS);
do
  echo $i
  $RECOMMENDER_EXPERIMENT_PYTHON_BIN __main__.py configs/global.json -n 0 -s $i
done

touch make-targets/global_analysis.completed

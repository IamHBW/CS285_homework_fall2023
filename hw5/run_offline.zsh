#!/usr/bin/env zsh

# Default to cql if no argument is provided
AGENT=${1:-cql}
echo "Running offline RL experiments with agent: $AGENT"

python ./cs285/scripts/run_hw5_offline.py \
-cfg experiments/offline/pointmass_easy_${AGENT}.yaml \
--dataset_dir datasets

python ./cs285/scripts/run_hw5_offline.py \
-cfg experiments/offline/pointmass_medium_${AGENT}.yaml \
--dataset_dir datasets

python ./cs285/scripts/run_hw5_offline.py \
-cfg experiments/offline/pointmass_hard_${AGENT}.yaml \
--dataset_dir datasets

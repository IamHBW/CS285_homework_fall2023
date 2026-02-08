#!/bin/zsh

echo "Running CQL experiment for dataset size 1000..."
python cs285/scripts/run_hw5_offline.py -cfg experiments/offline/pointmass_hard_cql_1000.yaml --dataset_dir datasets/

echo "Running CQL experiment for dataset size 5000..."
python cs285/scripts/run_hw5_offline.py -cfg experiments/offline/pointmass_hard_cql_5000.yaml --dataset_dir datasets/

echo "Running CQL experiment for dataset size 10000..."
python cs285/scripts/run_hw5_offline.py -cfg experiments/offline/pointmass_hard_cql_10000.yaml --dataset_dir datasets/

echo "Running CQL experiment for dataset size 20000..."
python cs285/scripts/run_hw5_offline.py -cfg experiments/offline/pointmass_hard_cql_20000.yaml --dataset_dir datasets/

echo "All CQL experiments completed."

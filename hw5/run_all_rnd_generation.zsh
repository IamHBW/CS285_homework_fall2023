#!/bin/zsh

echo "Generating dataset size 1000..."
python cs285/scripts/run_hw5_explore.py -cfg experiments/exploration/pointmass_hard_rnd_1000.yaml --dataset_dir datasets/

echo "Generating dataset size 5000..."
python cs285/scripts/run_hw5_explore.py -cfg experiments/exploration/pointmass_hard_rnd_5000.yaml --dataset_dir datasets/

echo "Generating dataset size 10000..."
python cs285/scripts/run_hw5_explore.py -cfg experiments/exploration/pointmass_hard_rnd_10000.yaml --dataset_dir datasets/

echo "Generating dataset size 20000..."
python cs285/scripts/run_hw5_explore.py -cfg experiments/exploration/pointmass_hard_rnd_20000.yaml --dataset_dir datasets/

echo "All datasets generated successfully."

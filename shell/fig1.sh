#!/bin/bash

# Define arrays for all parameters
seeds=(0 1 2 3 4)
methods=("FedAvg" "FedDyn" "FedSAM" "FedGamma" "FedSpeed" "FedSMOO")

# Run all combinations
for s in "${seeds[@]}"; do
    python train.py --model kan --method ${method} --layer 1 --seed ${s}
done

for s in "${seeds[@]}"; do
    python train.py --model mlp --method ${method} --layer 1 --seed ${s}
done

for s in "${seeds[@]}"; do
    python train.py --model mlp --method ${method} --layer 2 --seed ${s}
done

for s in "${seeds[@]}"; do
    python train.py --model mlp --method ${method} --layer 3 --seed ${s}
done
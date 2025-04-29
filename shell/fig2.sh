#!/bin/bash

# Define arrays for all parameters
seeds=(0 1 2 3 4)
methods=("FedAvg" "FedDyn" "FedSAM" "FedGamma" "FedSpeed" "FedSMOO")

# Run all combinations
for s in "${seeds[@]}"; do
    python train.py --model kan --noniid --method ${method} --layer 1 --split-coef 1.0 --seed ${s} 
done

for s in "${seeds[@]}"; do
    python train.py --model mlp --noniid --method ${method} --layer 1 --split-coef 1.0 --seed ${s}
done

for s in "${seeds[@]}"; do
    python train.py --model mlp --noniid --method ${method} --layer 2 --split-coef 1.0 --seed ${s}
done

for s in "${seeds[@]}"; do
    python train.py --model mlp --noniid --method ${method} --layer 3 --split-coef 1.0 --seed ${s}
done
#!/bin/bash

# Define arrays for all parameters
methods=("FedAvg" "FedDyn" "FedSAM" "FedGamma" "FedSpeed" "FedSMOO")
seeds=(0 1 2 3 4)

# Run all combinations
for s in "${seeds[@]}"; do
    python train.py --non-iid --model kan --method ${method} --layer 1 --comm-rounds 3000 --seed ${s}
done

# Run all combinations
for s in "${seeds[@]}"; do
    python train.py --non-iid --model kan --method ${method} --layer 3 --comm-rounds 3000 --seed ${s}
done

# Run all combinations
for s in "${seeds[@]}"; do
    python train.py --non-iid --model kan --method ${method} --layer 5 --comm-rounds 3000 --seed ${s}
done

# Run all combinations
for s in "${seeds[@]}"; do
    python train.py --non-iid --model kan --method ${method} --layer w3 --comm-rounds 3000 --seed ${s}
done

# Run all combinations
for s in "${seeds[@]}"; do
    python train.py --non-iid --model kan --method ${method} --layer w5 --comm-rounds 3000 --seed ${s}
done
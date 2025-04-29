#!/bin/bash

# Define arrays for all parameters
methods=("FedAvg" "FedDyn" "FedSAM" "FedGamma" "FedSpeed" "FedSMOO")
grids=(3 5 10)
coefs=(0.001 0.1 1.0 10.0)
seeds=(0 1 2 3 4)

# Run all combinations
for s in "${seeds[@]}"; do
    for method in "${methods[@]}"; do
        for g in "${grids[@]}"; do
            for c in "${coefs[@]}"; do
                python train.py --non-iid --model kan --method ${method} --grid ${g} --split-coef ${c} --layer 1 --seed ${s}
            done
        done
    done
done

for s in "${seeds[@]}"; do
    for method in "${methods[@]}"; do
        for g in "${grids[@]}"; do
            for c in "${coefs[@]}"; do
                python train.py --non-iid --model mlp --method ${method} --grid ${g} --split-coef ${c} --layer 1 --seed ${s}
            done
        done
    done
done

for s in "${seeds[@]}"; do
    for method in "${methods[@]}"; do
        for g in "${grids[@]}"; do
            for c in "${coefs[@]}"; do
                python train.py --non-iid --model mlp --method ${method} --grid ${g} --split-coef ${c} --layer 2 --seed ${s}
            done
        done
    done
done

for s in "${seeds[@]}"; do
    for method in "${methods[@]}"; do
        for g in "${grids[@]}"; do
            for c in "${coefs[@]}"; do
                python train.py --non-iid --model mlp --method ${method} --grid ${g} --split-coef ${c} --layer 3 --seed ${s}
            done
        done
    done
done
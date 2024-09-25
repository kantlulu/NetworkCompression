#!/usr/bin/env bash


# ------------------------
# NOTED: Need to replace the seed and --load-path
# We use seed 8152, 1011, 6162, 2177 in the experiments
# and load-path of each of them corresponds to pruned
# .pt model after running pruning.py by using the same seed
# ------------------------


# ------------------------
# CIFAR-10
# ------------------------
python3 quantize_encode.py --model resnet56 --dataset cifar10 --n-epochs 20 --lr 0.001 --quan-mode conv-quan --load-path saves/1625594011/model_best.pt --quan-bits 5 --seed 8152
python3 quantize_encode.py --model resnet56 --dataset cifar10 --n-epochs 20 --lr 0.001 --quan-mode conv-quan --load-path saves/1625594011/model_best.pt --quan-bits 8 --seed 8152


# ------------------------
# CIFAR-100
# ------------------------
python3 quantize_encode.py --model resnet56 --dataset cifar100 --n-epochs 20 --lr 0.0001 --quan-mode conv-quan --load-path saves/1625594199/model_best.pt --quan-bits 5 --seed 8152

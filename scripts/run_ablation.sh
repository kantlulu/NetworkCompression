#!/usr/bin/env bash

seeds=(8152 1011 6162 2177)


for seed in "${seeds[@]}"
do
    # RAND + HAP + KD
    python3 pruning.py --t-model resnet56 --s-copy-t --dataset cifar100 --prune-rates 0.6 --prune-mode filter-r --t-path saves/1625594199/model_best.pt --distill hap --betas 700 --hap-ts 3 --log-name ABLATION.txt --seed "$seed"

    # PFEC + HAP + KD
    python3 pruning.py --t-model resnet56 --s-copy-t --dataset cifar100 --prune-rates 0.6 --prune-mode filter-a --t-path saves/1625594199/model_best.pt --distill hap --betas 700 --hap-ts 3 --log-name ABLATION.txt --seed "$seed"

    # FPGM + HAP + KD
    python3 pruning.py --t-model resnet56 --s-copy-t --dataset cifar100 --prune-rates 0.6 --prune-mode filter-gm --t-path saves/1625594199/model_best.pt --distill hap --betas 700 --hap-ts 3 --log-name ABLATION.txt --seed "$seed"

    # NGGM + AT + KD
    python3 pruning.py --t-model resnet56 --s-copy-t --dataset cifar100 --prune-rates 0.6 --prune-mode filter-nggm --samp-batches 25 --t-path saves/1625594199/model_best.pt --distill at --betas 1000 --log-name ABLATION.txt --seed "$seed"

    # NGGM + SP + KD
    python3 pruning.py --t-model resnet56 --s-copy-t --dataset cifar100 --prune-rates 0.6 --prune-mode filter-nggm --samp-batches 25 --t-path saves/1625594199/model_best.pt --distill sp --betas 3000 --log-name ABLATION.txt --seed "$seed"

    # NGGM + HAP + KD
    python3 pruning.py --t-model resnet56 --s-copy-t --dataset cifar100 --prune-rates 0.6 --prune-mode filter-nggm --samp-batches 25 --t-path saves/1625594199/model_best.pt --distill hap --betas 700 --log-name ABLATION.txt --seed "$seed"
done


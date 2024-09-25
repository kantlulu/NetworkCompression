#!/usr/bin/env bash

seeds=(8152 1011 6162 2177)


# ------------------------
# CIFAR-10
# ------------------------
function cifar10 () {
    for seed in "${seeds[@]}"
    do
        # NGGM + HAP + KD
        python3 pruning.py --t-model resnet56 --s-copy-t --dataset cifar10 --prune-rates 0.6 --prune-mode filter-nggm --samp-batches 25 --t-path saves/1625594011/model_best.pt --distill hap --betas 700 --hap-ts 3 --log-name PRUNED-CIFAR10.txt --seed "$seed"

        # NGGM
        python3 pruning.py --t-model resnet56 --s-copy-t --dataset cifar10 --prune-rates 0.6 --prune-mode filter-nggm --samp-batches 25 --t-path saves/1625594011/model_best.pt --log-name PRUNED-CIFAR10.txt --seed "$seed"

        # AT + KD
        python3 pruning.py --t-model resnet56 --s-copy-t --dataset cifar10 --prune-rates 0.6 --prune-mode filter-r --t-path saves/1625594011/model_best.pt --distill at --betas 1000 --log-name PRUNED-CIFAR10.txt --seed "$seed"

        # SP + KD
        python3 pruning.py --t-model resnet56 --s-copy-t --dataset cifar10 --prune-rates 0.6 --prune-mode filter-r --t-path saves/1625594011/model_best.pt --distill sp --betas 3000 --log-name PRUNED-CIFAR10.txt --seed "$seed"

        # AFD + KD
        python3 pruning.py --t-model resnet56 --s-copy-t --dataset cifar10 --prune-rates 0.6 --prune-mode filter-r --t-path saves/1625594011/model_best.pt --distill afd --betas 200 --log-name PRUNED-CIFAR10.txt --seed "$seed"

        # PFEC
        python3 pruning.py --t-model resnet56 --s-copy-t --dataset cifar10 --prune-rates 0.6 --prune-mode filter-a --t-path saves/1625594011/model_best.pt --log-name PRUNED-CIFAR10.txt --seed "$seed"

        # SFP
        python3 pruning.py --t-model resnet56 --s-copy-t --dataset cifar10 --prune-rates 0.6 --prune-mode filter-a --soft-prune --prune-interval 1 --t-path saves/1625594011/model_best.pt --log-name PRUNED-CIFAR10.txt --seed "$seed"

        # FPGM
        python3 pruning.py --t-model resnet56 --s-copy-t --dataset cifar10 --prune-rates 0.6 --prune-mode filter-gm --t-path saves/1625594011/model_best.pt --log-name PRUNED-CIFAR10.txt --seed "$seed"
    done
}


# ------------------------
# CIFAR-100
# ------------------------
function cifar100 () {
    for seed in "${seeds[@]}"
    do
        # NGGM + HAP + KD
        python3 pruning.py --t-model resnet56 --s-copy-t --dataset cifar100 --prune-rates 0.6 --prune-mode filter-nggm --samp-batches 25 --t-path saves/1625594199/model_best.pt --distill hap --betas 700 --hap-ts 3 --log-name PRUNED-CIFAR100.txt --seed "$seed"

        # NGGM
        python3 pruning.py --t-model resnet56 --s-copy-t --dataset cifar100 --prune-rates 0.6 --prune-mode filter-nggm --samp-batches 25 --t-path saves/1625594199/model_best.pt --log-name PRUNED-CIFAR100.txt --seed "$seed"

        # AT + KD
        python3 pruning.py --t-model resnet56 --s-copy-t --dataset cifar100 --prune-rates 0.6 --prune-mode filter-r --t-path saves/1625594199/model_best.pt --distill at --betas 1000 --log-name PRUNED-CIFAR100.txt --seed "$seed"

        # SP + KD
        python3 pruning.py --t-model resnet56 --s-copy-t --dataset cifar100 --prune-rates 0.6 --prune-mode filter-r --t-path saves/1625594199/model_best.pt --distill sp --betas 3000 --log-name PRUNED-CIFAR100.txt --seed "$seed"

        # AFD + KD
        python3 pruning.py --t-model resnet56 --s-copy-t --dataset cifar100 --prune-rates 0.6 --prune-mode filter-r --t-path saves/1625594199/model_best.pt --distill afd --betas 200 --log-name PRUNED-CIFAR100.txt --seed "$seed"

        # PFEC
        python3 pruning.py --t-model resnet56 --s-copy-t --dataset cifar100 --prune-rates 0.6 --prune-mode filter-a --t-path saves/1625594199/model_best.pt --log-name PRUNED-CIFAR100.txt --seed "$seed"

        # SFP
        python3 pruning.py --t-model resnet56 --s-copy-t --dataset cifar100 --prune-rates 0.6 --prune-mode filter-a --soft-prune --prune-interval 1 --t-path saves/1625594199/model_best.pt --log-name PRUNED-CIFAR100.txt --seed "$seed"

        # FPGM
        python3 pruning.py --t-model resnet56 --s-copy-t --dataset cifar100 --prune-rates 0.6 --prune-mode filter-gm --t-path saves/1625594199/model_best.pt --log-name PRUNED-CIFAR100.txt --seed "$seed"
   done
}


# ------------------------
# CINIC-10
# ------------------------
function cinic10 () {
    for seed in "${seeds[@]}"
    do
        # NGGM + HAP + KD
        python3 pruning.py --t-model resnet56 --s-copy-t --t-path saves/1630871975/model_best.pt --dataset cinic10 --prune-rates 0.6 --prune-mode filter-nggm --samp-batches 25 --distill hap --alpha 0.6 --betas 700 --kd-t 16.0 --hap-ts 3 --n-epochs 140 --lr 0.01 --schedule 100 120 --lr-drops 0.1 0.1 --batch-size 96 --log-name PRUNED-CINIC10.txt --seed "$seed"

        # NGGM
        python3 pruning.py --t-model resnet56 --s-copy-t --t-path saves/1630871975/model_best.pt --dataset cinic10 --prune-rates 0.6 --prune-mode filter-nggm --samp-batches 25 --n-epochs 140 --lr 0.01 --schedule 100 120 --lr-drops 0.1 0.1 --batch-size 96 --log-name PRUNED-CINIC10.txt --seed "$seed"

        # AT + KD
        python3 pruning.py --t-model resnet56 --s-copy-t --t-path saves/1630871975/model_best.pt --dataset cinic10 --prune-rates 0.6 --prune-mode filter-r --distill at --alpha 0.6 --betas 50 --kd-t 16.0 --n-epochs 140 --lr 0.01 --schedule 100 120 --lr-drops 0.1 0.1 --batch-size 96 --log-name PRUNED-CINIC10.txt --seed "$seed"

        # SP + KD
        python3 pruning.py --t-model resnet56 --s-copy-t --t-path saves/1630871975/model_best.pt --dataset cinic10 --prune-rates 0.6 --prune-mode filter-r --distill sp --alpha 0.6 --betas 2000 --kd-t 16.0 --n-epochs 140 --lr 0.01 --schedule 100 120 --lr-drops 0.1 0.1 --batch-size 96 --log-name PRUNED-CINIC10.txt --seed "$seed"

        # PFEC
        python3 pruning.py --t-model resnet56 --s-copy-t --t-path saves/1630871975/model_best.pt --dataset cinic10 --prune-rates 0.6 --prune-mode filter-a --n-epochs 140 --lr 0.01 --schedule 100 120 --lr-drops 0.1 0.1 --batch-size 96 --log-name PRUNED-CINIC10.txt --seed "$seed"

        # SFP
        python3 pruning.py --t-model resnet56 --s-copy-t --t-path saves/1630871975/model_best.pt --dataset cinic10 --prune-rates 0.6 --prune-mode filter-a --soft-prune --prune-interval 1 --n-epochs 140 --lr 0.01 --schedule 100 120 --lr-drops 0.1 0.1 --batch-size 96 --log-name PRUNED-CINIC10.txt --seed "$seed"

        # FPGM
        python3 pruning.py --t-model resnet56 --s-copy-t --t-path saves/1630871975/model_best.pt --dataset cinic10 --prune-rates 0.6 --prune-mode filter-gm --n-epochs 140 --lr 0.01 --schedule 100 120 --lr-drops 0.1 0.1 --batch-size 96 --log-name PRUNED-CINIC10.txt --seed "$seed"
    done
}


cifar10
cifar100
cinic10

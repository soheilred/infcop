#!/bin/bash

# touch ../output/errors/batch-run.out

LOG="../output/errors/batch-run.out"
echo "#############################" >> $LOG
date >> $LOG
echo $1 >> $LOG

ARCHS=("alexnet" "vgg16" "resnet18")
DATASETS=("IMAGENET" "CIFAR10" "MNIST") # 
for arch in ${ARCHS[@]}; do for dataset in ${DATASETS[@]}; do
    echo $arch $dataset >> ../output/errors/batch-run.out
    sbatch experiments.slurm --lr=.001 --arch=$arch --dataset=$dataset \
        --train_epochs=3 --control_at_iter=-1 --control_at_epoch=2 \
        --control_at_layer="2" --experiment_type=$1 \
        --acc_thrd=90 >> $LOG

    sbatch experiments.slurm --lr=.001 --arch=$arch --dataset=$dataset \
        --train_epochs=3 --control_at_iter=1 --control_at_epoch=2 \
        --control_at_layer="2" --experiment_type=$1 \
        --acc_thrd=90 >> $LOG
done; done;
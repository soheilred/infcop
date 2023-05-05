#!/bin/bash
#SBATCH --job-name=compute_correlation  # Job name
#SBATCH --partition=grtx                # Select partition
#SBATCH --cpus-per-task=4               # Run on a 4 cores per node
#SBATCH --nodes=1                       # Run on a 1 node
#SBATCH --ntasks=6                      # One process (shared-memory)
#SBATCH --mem=16gb                     # Job memory request
#SBATCH --time=1000:00:00               # Time limit hrs:min:sec
#SBATCH --gres=gpu:6                    # Request one GPU
#SBATCH --gpus-per-task=1               # Run on 1 gpu per node
#SBATCH --output=../output/errors/tbst_%j.log   # Standard output and error log
#SBATCH --error=../output/errors/error_%j
#SBATCH --mail-type=BEGIN,END,FAIL          # Mail events
#SBATCH --mail-user=soheil.gharatappeh@maine.edu


echo "====================================================="
pwd; hostname; date
echo "====================================================="

# SBATCH --partition=grtx                # Select partition
# SBATCH --partition=dgx                # Select partition
# module load gnu8/8.3.0 mvapich2/2.3.2  /21.08
echo "Loading modules"
module load singularity nv/pytorch anaconda3 
echo "Modules loaded"

. $CONDA_BIN/conda-init
conda activate ib

ARCHS=("alexnet" "vgg16" "resnet")
DATASETS=("IMAGENET" "CIFAR10" "MNIST")
for arch in ${ARCHS[@]}; do for dataset in ${DATASETS[@]}; do
    echo $arch $dataset
    srun -n 1 singularity run --nv /home/gharatappeh/images/pytorch.sif python ~/nips/src/prune.py --lr=.001 --arch=$arch --dataset=$dataset --train_epochs=3 --control_at_iter=1 --control_at_epoch=-1 --control_at_layer="2" &
    srun -n 1 singularity run --nv /home/gharatappeh/images/pytorch.sif python ~/nips/src/prune.py --lr=.001 --arch=$arch --dataset=$dataset --train_epochs=3 --control_at_iter=1 --control_at_epoch=2 --control_at_layer="2" &
done; done;
wait
#+end_src
echo "====================================================="
date


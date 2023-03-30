#!/bin/bash
#SBATCH --job-name=compute_correlation  # Job name
#SBATCH --partition=dgx                # Select partition, grtx, dgx
#SBATCH --cpus-per-task=1               # Run on a 4 cores per node
#SBATCH --nodes=1                       # Run on a 1 node
#SBATCH --ntasks=1                      # One process (shared-memory)
#SBATCH --mem=128gb                     # Job memory request
#SBATCH --time=1000:00:00               # Time limit hrs:min:sec
#SBATCH --gres=gpu:1                    # Request one GPU
#SBATCH --output=/home/gharatappeh/ib_dl/output/errors/tbst_%j.log   # Standard output and error log
#SBATCH --error=/home/gharatappeh/ib_dl/output/errors/error_%j
#SBATCH --mail-type=BEGIN,END,FAIL          # Mail events
#SBATCH --mail-user=soheil.gharatappeh@maine.edu


echo "====================================================="
pwd; hostname; date
echo "====================================================="

# module load gnu8/8.3.0 mvapich2/2.3.2  /21.08
echo "Loading modules"
module load singularity nv/pytorch anaconda3 
echo "Modules loaded"

. $CONDA_BIN/conda-init
conda activate ib

# time srun singularity run ~/pytorch-sing.simg python correlation.py train
# singularity run ~/pytorch-sing.simg python correlation.py correlation
# singularity exec --nv ~/pytorch-sing.sif python ~/ib_dl/src/correlation.py train
# singularity run --nv ~/images/pytorch.sif python ~/ib_dl/src/correlation.py train
# singularity run --nv ~/images/pytorch.sif python ~/ib_dl/src/experiments.py
singularity run --nv /home/gharatappeh/images/pytorch.sif python "$@"
# singularity run --nv ~/images/pytorch.sif python ~/ib_dl/src/test.py
# singularity run --nv ~/images/bare.simg python ~/ib_dl/src/correlation.py train
# srun python ~/ib_dl/src/correlation.py train
echo "====================================================="
date


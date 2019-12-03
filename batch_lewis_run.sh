#!/bin/bash

#SBATCH --partition gpu3,gpu4
#SBATCH -A rc-gpu
#SBATCH --cpus-per-task=1                  # number of cores
#SBATCH --mem=4G                           # total memory
#SBATCH --gres gpu:1
#SBATCH --job-name=ca1
#SBATCH --output=ca1%j.out
#SBATCH --time 0-02:30

module load cuda/cuda-9.0.176
module load cudnn/cudnn-7.1.4-cuda-9.0.176

source activate tf_gpu

echo "Running model at $(date)"

python LFP_Classification_BL.py

echo "Done running model at $(date)"

source deactivate

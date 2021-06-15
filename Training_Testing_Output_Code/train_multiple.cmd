#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -t 72:00:00
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=jl40@princeton.edu
#SBATCH --output=60s.out
#SBATCH --mem=250G

module purge
module load anaconda3
module load cudnn/cuda-10.1
module list

env | grep SLURM

source activate tf2-gpu
python run_multiple.py

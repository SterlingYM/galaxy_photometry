#!/bin/bash

#SBATCH --job-name=SersicPhot
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --time=30:00:00
#SBATCH --partition=shared

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=4000

source ~/.bashrc
source activate sphot
python3 test_SersicFit.py 

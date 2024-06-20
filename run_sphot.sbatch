#!/bin/bash
#SBATCH --job-name=sphot
#SBATCH --array=0-200 # Job array range
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --time=3:00:00
#SBATCH --partition=parallel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=6400

source ~/.bashrc
source activate sphot
FILE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" test_Jun18.txt)
srun python3 run_sphot.py $FILE --out_folder=sphot_out/Jun18_tests/
# srun python3 run_sphot.py sphot_out/May29/g144_sphot.h5 --scalefit_only


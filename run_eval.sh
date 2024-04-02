#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 03:00:00
#SBATCH --gpus=v100-32:1

#type 'man sbatch' for more information and options
#this job will ask for 4 V100 GPUs on a v100-32 node in GPU-shared for 5 hours
#this job would potentially charge 20 GPU SUs

# move to working directory
# this job assumes:
# - all input data is stored in this directory
# - all output should be stored in this directory
# - please note that groupname should be replaced by your groupname
# - username should be replaced by your username
# - path-to-directory should be replaced by the path to your directory where the executable is

#run pre-compiled program which is already in your project space
date
# echo

# cd /jet/home/zyou2/BioLaySumm
SCRIPT_PATH=/jet/home/zyou2/BioLaySumm/evaluate_fine_tuned_model.py 

source activate
conda deactivate
conda activate biosum

# run a pre-compiled program which is already in your project space
CUDA_VISIBLE_DEVICES=0 python $SCRIPT_PATH
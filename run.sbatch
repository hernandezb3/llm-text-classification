#!/bin/bash
#SBATCH --job-name=llama_classify
#SBATCH --partition=general-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=brittney.hernandez@uconn.edu

# create logs dir if it doesn't exist
mkdir -p logs

# load modules
module purge
module load python/3.12.2

pip3 install -r requirements_hpc.txt

# run the script
python local.py


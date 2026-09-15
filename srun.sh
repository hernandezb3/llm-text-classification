
# shell into uconn hpc
ssh -Y netid@hpc2.storrs.hpc.uconn.edu

# srun (interactive) request
# use debug partition to test -- 30 min max
# general-gpu
srun --ntasks=1 --nodes=1 --partition=debug --pty bash

srun --partition=general-gpu \
     --nodes=1 \
     --ntasks=1 \
     --cpus-per-task=8 \
     --gres=gpu:1 \
     --mem=32G \
     --time=06:00:00 \
     --pty bash

# bigger models need multiple gpus
srun --partition=general-gpu \
     --nodes=1 \
     --ntasks=1 \
     --cpus-per-task=8 \
     --gres=gpu:2 \
     --mem=64G \
     --time=06:30:00 \
     --pty bash


# check which node was assigned
hostname

# list files in directory
ls

# *** only need to do 1-6 this once ***
# install virtual environment
# when you run conda init bash, you will need to close the session and shell in again
curl -L -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Miniconda3-latest-Linux-x86_64.sh # 1
bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3 # 2
export PATH=$HOME/miniconda3/bin:$PATH # 3
# google colab runs on 3.12.3 
conda create -n aimecon python=3.12.3 # 4
conda init bash # 5

# clone github
git clone https://github.com/hernandezb3/llm-text-classification.git # 6


# activate the virtual environment
conda activate aimecon

# after the repository is cloned, grab any changes that have been made using git pull
# change directory -- make sure you're in the repo
cd llm-text-classification
# make sure requirements.txt is in this folder
python3 -m pip install -r requirements_hpc.txt # only need to do once per virtual environment
# update changes
git pull

# run the script
python3 local.py

# to run as an sbatch request:
sbatch run.sbatch
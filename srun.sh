
# shell into uconn hpc
ssh -Y netid@hpc2.storrs.hpc.uconn.edu

# clone github
git clone https://github.com/hernandezb3/llm-text-classification.git

# change directory
cd llm-text-classification
# list files in directory
# make sure running from where requirements.txt and local.py are
ls

# srun (interactive) request
srun --partition=general-gpu \
     --nodes=1 \
     --ntasks=1 \
     --cpus-per-task=8 \
     --gres=gpu:1 \
     --mem=32G \
     --time=00:30:00 \
     --pty bash


# check which node was assigned
hostname

# google colab runs on 3.12.3 
module avail python
module purge
module load python/3.12.2

# install requirements directly to desired version of python
pip install --upgrade pip
python3 -m pip install -r requirements_hpc.txt

# run the script
python3 local.py

# update changes
git pull

# to run as an sbatch request:
sbatch run.sbatch
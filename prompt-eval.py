import os
import re
import time

from dotenv import load_dotenv
from pathlib import Path
from huggingface_hub import login
import torch
import pandas as pd
from pydantic import BaseModel
from enum import Enum
from tqdm import tqdm
import outlines
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# FILE STRUCTURE FOR HPC/COLAB
# .env in cd
# data to data/
# prompt_codebook to data_management/
# classifications.xlsx to results/
# make sure results/local exists

load_dotenv()

USER = "brittney" 

if USER == "brittney":
    WORKING_DIR = Path("/Users/brittneyhernandez/Library/CloudStorage/OneDrive-UniversityofConnecticut/AIME-con")
    DATA_DIR = WORKING_DIR / "data"
elif USER == "hpc":
    WORKING_DIR = Path.cwd()
    DATA_DIR =  WORKING_DIR / "data"
elif USER =="colab":
    from google.colab import drive
    drive.mount('/content/drive/')
    WORKING_DIR = Path.cwd()
    DATA_DIR = WORKING_DIR / "data"

RESULTS_DIR = WORKING_DIR / "results"
DATA_SOURCE = "train" # train, validate, test

path_to_prompts = WORKING_DIR / "data_management" / "empirical_prompts_50.csv"
prompts = pd.read_csv(path_to_prompts)
prompt_ids = prompts["prompt_id"]
prompt_conditions = prompt_ids.str.split("_").str[1]

# create dataframe
context_vars = [f"c{i}" for i in range(1,7)]
task_vars = [f"c{i}" for i in range(1,6)]
definition_vars = [f"c{i}" for i in range(1,5)]

pattern = r"(c[0-9])(t[0-9])(d[0-9])(g[0-9])"
ids = re.search()
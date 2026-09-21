import os
import re
import time
import random

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
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- get prompt codebook ----
path_to_prompts = WORKING_DIR / "data_management" / "prompt_variants.xlsx"

number_of_prompts = 5
prompt_variant_id = []

for i in range(0, number_of_prompts):
    prompt_dictionary = {}

    # randomly select 1 context variant
    context_sheet = pd.read_excel(path_to_prompts, sheet_name = "context")
    max_context = max(context_sheet.index)
    n_context = 1
    context_row = random.randint(0, max_context)
    prompt_dictionary["context"] = context_sheet.loc[context_row, "variant"]

    # randomly select 1 task variant
    task_sheet = pd.read_excel(path_to_prompts, sheet_name = "task")
    max_task = max(task_sheet.index)
    n_task = 1
    task_row = random.randint(0, max_task)
    prompt_dictionary["task"] = task_sheet.loc[task_row, "variant"]

    # randomly select one construct definition variant
    definition_sheet = pd.read_excel(path_to_prompts, sheet_name = "definition")
    max_definition = max(definition_sheet.index)
    n_definition = 1
    definition_row = random.randint(0, max_definition)
    prompt_dictionary["definition"] = definition_sheet.loc[definition_row, "variant"]

    # randomly draw 0-n guidance
    guidance_sheet = pd.read_excel(path_to_prompts, sheet_name = "guidance")
    total_guidance = guidance_sheet.shape[0]
    n_guidance = random.randint(0, total_guidance)
    guidance_rows = random.sample(range(0, total_guidance), n_guidance)
    guidance_rows.sort()


    guidance = ""
    for row in guidance_rows:
        guidance_row = guidance_sheet.loc[row, "variant"]
        guidance = guidance + guidance_row + "\n"

    prompt_dictionary["guidance"] = guidance


    format_sheet = pd.read_excel(path_to_prompts, sheet_name = "format")
    prompt_dictionary["format_local"] = format_sheet.loc[format_sheet.id == "format_local", "variant"].item()
    prompt_dictionary["format_case"] = format_sheet.loc[format_sheet.id == "format_case", "variant"].item()
    prompt_dictionary["format_guidance"] = format_sheet.loc[format_sheet.id == "format_guidance", "variant"].item()

prompt_manifest = []




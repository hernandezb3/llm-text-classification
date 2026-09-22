import os

from dotenv import load_dotenv
from pathlib import Path
from huggingface_hub import login
import torch
import pandas as pd
import outlines
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

# FILE STRUCTURE FOR HPC/COLAB
# .env in cd
# data to data/
# prompt_codebook to data_management/
# classifications.xlsx to results/
# make sure results/local exists

load_dotenv()

USER = "local" 

if USER == "local":
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

print(f"\nUsing device: {DEVICE}")
if DEVICE == "cuda":
    print(torch.cuda.get_device_name(0))

# ---- get data ----
data_filename = f"cgi_{DATA_SOURCE}"
path_to_data = DATA_DIR / f"{data_filename}.xlsx"
df = pd.read_excel(path_to_data)
#df = df.sample(n = 5, ignore_index = True)
# call out in the room, what performance did you estimate
# performance metrics are estimates > seguey to uncertainty

# ---- get prompt codebook ----
path_to_prompts = WORKING_DIR / "data_management" / "llm_codebook.xlsx"
prompts = pd.read_excel(path_to_prompts)

prompt_dictionary = prompts.to_dict()
         
# create a function to add a case to the prompt
def prompt_case(case, p):
    parts = [
        p["task_1"],
        p["definition_1"],
        p["format_case"],
    ]

    parts.append(f"\n\"\"\"{case}\"\"\"\n")                 # <- the case goes here
    parts.append(p["format_local"])    # e.g. output format instructions

    return "\n\n".join(part for part in parts if part)

# ---- set up model ----
login(token = os.getenv("HF_TOKEN"))

MODEL = "meta-llama/Llama-3.2-1B-Instruct"
TASK = "text-generation"
TOKENS = 500
TEMPERATURE = 0.1
PRECISION = torch.bfloat16 # can use bfloat16 or bfloat32 if cuda is available (float for cpu, bfloat for gpu)

print(f"Model: {MODEL}\nSample Size: {df.shape[0]}\n")

# make sure permissions are on 
#client = transformers.pipeline(TASK, model = MODEL, 
#                               model_kwargs = {"dtype": PRECISION}, 
#                               token = os.getenv("HF_TOKEN")) # initate pipeline

model = AutoModelForCausalLM.from_pretrained(MODEL, 
                                              dtype = PRECISION,
                                              token = os.getenv("HF_TOKEN"),
                                              device_map = DEVICE) # initate pipeline

hf_tokenizer = AutoTokenizer.from_pretrained(MODEL, token = os.getenv("HF_TOKEN"))

client = outlines.from_transformers(model, hf_tokenizer)

code_local = []
explanation_local = []

tp = 0
tn = 0
fp = 0
fn = 0
format_errors = 0

# format response
pattern = r"```(yes|no)```"
label_map = {"yes": 1, "no": 0}

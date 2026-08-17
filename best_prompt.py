import os

from dotenv import load_dotenv
from huggingface_hub import login
import transformers
import torch
import openpyxl
import pandas as pd

load_dotenv()

USER_PATH = os.getenv("USER_PATH")
GITHUB_PATH = USER_PATH + "Documents/github"
ONEDRIVE_PATH = USER_PATH + "Library/CloudStorage/OneDrive-UniversityofConnecticut/"
AIMECON_PATH = ONEDRIVE_PATH + "AIME-con/"
FOCUS_PATH = ONEDRIVE_PATH + "focus/project focus/focus_main/"


# ---- get prompt codebook ----
path_to_prompts = AIMECON_PATH + "data_management/"
prompts = pd.read_excel(path_to_prompts + "prompt_codebook.xlsx")

PROMPT = prompts.loc[prompts.id == "P1", "prompt"].item() + prompts.loc[prompts.id == "P2", "prompt"].item()


# ---- get data ----
path_to_data = FOCUS_PATH + "data/prompt_codes/cgi/clean/"
df = pd.read_excel(path_to_data + "cgi_full_data.xlsx")
DATA = df.loc[0, "Text"]



# ---- set up model ----

login(token = os.getenv("HF_TOKEN"))

MODEL = "meta-llama/Llama-3.2-1B-Instruct"
TASK = "text-generation"
TOKENS = 100
TEMPERATURE = 0.2
QUANTIZATION = torch.bfloat16

weights = {"torch_dtype": QUANTIZATION} 

# make sure permissions are on 
pipeline = transformers.pipeline(TASK, model = MODEL, model_kwargs = weights, token = os.getenv("HF_TOKEN")) # initate pipeline

prompt_case = PROMPT + " " + DATA
input = [{"role": "user", "content": prompt_case}]
output = pipeline(input, max_new_tokens = TOKENS)
response = output[0]["generated_text"][-1]["content"]

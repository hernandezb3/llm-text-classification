import os
import re
import time

from dotenv import load_dotenv
from pathlib import Path
from huggingface_hub import login
import transformers
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from pydantic import BaseModel
from enum import Enum


load_dotenv()

FOCUS_DIR = Path("/Users/brittneyhernandez/Library/CloudStorage/OneDrive-UniversityofConnecticut/focus/project focus/focus_main")
AIMECON_DIR = Path("/Users/brittneyhernandez/Library/CloudStorage/OneDrive-UniversityofConnecticut/AIME-con")
FOCUS_DATA_DIR = FOCUS_DIR / "data/prompt_codes/cgi"
AIMECON_DATA_DIR = AIMECON_DIR / "data"
RESULTS_DIR = AIMECON_DIR / "results"
DATA_SOURCE = "train" # train, validate, test

# ---- get prompt codebook ----
path_to_prompts = AIMECON_DIR / "data_management" / "prompt_codebook.xlsx"
prompts = pd.read_excel(path_to_prompts)


# ---- get data ----
data_filename = f"cgi_{DATA_SOURCE}"
path_to_data = AIMECON_DATA_DIR / f"{data_filename}.xlsx"
df = pd.read_excel(path_to_data)
df = df.sample(n = 30, ignore_index = True)
# call out in the room, what performance did you estimate
# performance metrics are estimates > seguey to uncertainty


# ---- set up model ----
login(token = os.getenv("HF_TOKEN"))

MODEL = "meta-llama/Llama-3.2-1B-Instruct"
TASK = "text-generation"
TOKENS = 100
TEMPERATURE = 0.1
QUANTIZATION = torch.bfloat16


# make sure permissions are on 
client = transformers.pipeline(TASK, model = MODEL, 
                               model_kwargs = {"torch_dtype": QUANTIZATION} , 
                               token = os.getenv("HF_TOKEN")) # initate pipeline

code_local = []
explanation_local = []

tp = 0
tn = 0
fp = 0
fn = 0

# format response
pattern = r"```(yes|no)```"
label_map = {"yes": 1, "no": 0}

start = time.perf_counter() # start runtime counter

# FOR TESTING
each_case = 0
for row in df.index:
    CASE = df.loc[row, "text"]

    PROMPT = (prompts.loc[prompts.id == "Coding", "prompt"].item() + 
                  prompts.loc[prompts.id == "Construct", "prompt"].item() +
                  prompts.loc[prompts.id == "Prompt1", "prompt"].item() +
                  f"\"\"\"{CASE}\"\"\"" + 
                  prompts.loc[prompts.id == "Format2", "prompt"].item()
                  )

    prompt = [{"role": "user", "content": PROMPT}]

    prompt_local = client(
        prompt,
        temperature = TEMPERATURE,
        do_sample = True,  # temperature has no effect unless do_sample=True
        )
    response = prompt_local[0]["generated_text"][-1]["content"]

    structured_response = re.search(pattern, response)
    if structured_response:
        response_code = label_map[structured_response.group(1)]
    else:
        print(f"no regex pattern detected in response: {response}")
        response_code = None

    code_local.append(response_code)
    explanation_local.append(response)

    if df.loc[row, "code_human"] == 1 and response_code == 1:
        tp = tp + 1
    elif df.loc[row, "code_human"] == 0 and response_code == 0:
        tn = tn + 1
    elif df.loc[row, "code_human"] == 0 and response_code == 1:
        fp = fp + 1
    elif df.loc[row, "code_human"] == 1 and response_code == 0:
        fn = fn + 1

end = time.perf_counter()

df["code_llm"] = code_local




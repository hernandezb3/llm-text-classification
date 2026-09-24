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
DATA_SOURCE = "dev" # train, dev, test
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\nUsing device: {DEVICE}")
if DEVICE == "cuda":
    print(torch.cuda.get_device_name(0))

# ---- get data ----
path_to_data = DATA_DIR / f"{DATA_SOURCE}.xlsx"
df = pd.read_excel(path_to_data)
#df = df.sample(n = 5, ignore_index = True)
# call out in the room, what performance did you estimate
# performance metrics are estimates > seguey to uncertainty


# ---- get prompt codebook ----
path_to_prompts = WORKING_DIR / "data_management" / "llm_codebook.xlsx"
prompts = pd.read_excel(path_to_prompts)


# ---- set up model ----
login(token = os.getenv("HF_TOKEN"))

# ON HPC
# single gpu < 30B
# meta-llama/Llama-3.2-1B-Instruct (baseline) x
# meta-llama/Llama-3.2-3B-Instruct
# meta-llama/Llama-3.1-8B-Instruct
# meta-llama/Llama-4-Maverick-17B-128E-Instruct
# Qwen/Qwen2.5-7B-Instruct x
# google/gemma-4-12B-it x
# ibm-granite/granite-4.2-8b 
# microsoft/phi-4 (15B) x

# multi-gpus > 30B
# meta-llama/Llama-3.3-70B-Instruct
# Qwen/Qwen3-30B-A3B-Instruct-2507
# deepseek-ai/DeepSeek-V3.2
# google/gemma-4-31B-it


# ON COLAB
# meta-llama/Llama-3.2-1B-Instruct (baseline) x
path_to_finetuned_model = RESULTS_DIR / "finetune" / "Llama-3.2-1B-Instruct_dialogue_tuned"
MODEL = path_to_finetuned_model
TASK = "text-generation"
TOKENS = 500
TEMPERATURE = 0.1
PRECISION = torch.bfloat16 # can use bfloat16 or bfloat32 if cuda is available (float for cpu, bfloat for gpu)

print(f"Model: {MODEL}\nSample Size: {df.shape[0]}\n")

# format output
class Answer(str, Enum):
    yes = "yes"
    no = "no"

class Classification(BaseModel):
    explanation: str  # comes first, so reasoning happens before the label
    label: Answer


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

start = time.perf_counter() # start runtime counter

# FOR TESTING
each_case = 0
for row in tqdm(df.index):
    CASE = df.loc[row, "text"]

    PROMPT = (prompts.loc[prompts.id == "task_1", "prompt"].item() + " " +
                  prompts.loc[prompts.id == "definition_1", "prompt"].item() + " " +
                  prompts.loc[prompts.id == "format_case", "prompt"].item() + " " +
                  f"\n\"\"\"{CASE}\"\"\"\n" + " " +
                  prompts.loc[prompts.id == "format_local", "prompt"].item()
                  )

    if row == 0:
        print(f"\n{PROMPT}\n")

    prompt_local = client(PROMPT, Classification, max_new_tokens = TOKENS)

    try:
        parsed = Classification.model_validate_json(prompt_local)
        response_code = label_map[parsed.label.value]
        explanation_text = parsed.explanation
    except Exception as e:
        response_code = None
        explanation_text = prompt_local
        format_errors = format_errors + 1

    code_local.append(response_code)
    explanation_local.append(explanation_text)


    if df.loc[row, "code_human"] == 1 and response_code == 1:
        tp = tp + 1
    elif df.loc[row, "code_human"] == 0 and response_code == 0:
        tn = tn + 1
    elif df.loc[row, "code_human"] == 0 and response_code == 1:
        fp = fp + 1
    elif df.loc[row, "code_human"] == 1 and response_code == 0:
        fn = fn + 1

end = time.perf_counter()

df[f"code_{MODEL}"] = code_local
df[f"explanation_{MODEL}"] = explanation_local


# ---- save the results ----
# classifications
model_string = str(MODEL)
if "/" in model_string:
    last_split = len(re.split("/", model_string)) - 1
    model_stripped = re.split("/", model_string)[last_split]
else:
    model_stripped = MODEL

results_data_file = f"{data_filename}_{model_stripped}.xlsx"
path_to_data_results = RESULTS_DIR / "local" / results_data_file


# model performance
path_to_model_results = RESULTS_DIR / "classification.xlsx"
results = pd.read_excel(path_to_model_results, sheet_name = f"{DATA_SOURCE}")

new_row = {"model": model_stripped,
           "utterances": df.shape[0],
           "tp": tp,
           "tn": tn,
           "fp": fp,
           "fn": fn,
           "format_errors": format_errors,
           "cost": None,
           "runtime": end - start
           }

results = pd.concat([results, pd.DataFrame([new_row])], ignore_index = True)

try:
    df.to_excel(path_to_data_results, index = False)
    with pd.ExcelWriter(
        path_to_model_results, engine = "openpyxl", mode = "a", if_sheet_exists = "replace") as writer:
        results.to_excel(writer, sheet_name=f"{DATA_SOURCE}", index = False)
except:
    path_to_data_results = Path.cwd() / results_data_file
    df.to_excel(path_to_data_results, index = False)

    path_to_model_results = Path.cwd() / "classification.xlsx"
    mode = "a" if os.path.exists(path_to_model_results) else "w"
    if_sheet_exists = "replace" if mode == "a" else None

    with pd.ExcelWriter(
        path_to_model_results,
        engine = "openpyxl",
        mode = mode,
        if_sheet_exists = if_sheet_exists
        ) as writer:
            results.to_excel(writer, sheet_name = f"{DATA_SOURCE}", index = False)



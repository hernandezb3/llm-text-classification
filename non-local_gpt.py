import os
import time

from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

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
df = df.sample(n = 5, ignore_index = True)

# ---- set model params ----
GPT_MODEL = "gpt-5.6-terra" # https://developers.openai.com/api/docs/models/all

IN_RATE = 2.00
OUT_RATE = 12.00

# ---- prompt GPT ----
# get api key: https://platform.openai.com/api-keys

# check that the api key got loaded in the .env
# if loaded, prints the key
# if not loaded, prints ERROR
key = "OPENAI_API_KEY"
print(os.environ.get(key, f"ERROR: Variable {key} Not Found"))

# initialize model
openai = OpenAI()

# initialize output objects
tokens_in_column = []
tokens_out_column = []

total_input_cost = None
total_output_cost = None

code_gpt = []

tp = 0
tn = 0
fp = 0
fn = 0

start = time.perf_counter() # start runtime counter

# FOR TESTING
row = 0
for row in df.index:
    CASE = df.loc[row, "text"]

    # construct prompt w case
    PROMPT = (prompts.loc[prompts.id == "Coding", "prompt"].item() + 
              prompts.loc[prompts.id == "Construct", "prompt"].item() +
              prompts.loc[prompts.id == "Prompt1", "prompt"].item() +
              f"\"{CASE}\"" + 
              prompts.loc[prompts.id == "Format", "prompt"].item()
              )

    # format prompt for gpt
    prompt = [
        #{"role": "system", "content": CONTEXT},
        {"role": "user", "content": PROMPT}
        ]

    # model settings
    # https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
    prompt_gpt = openai.chat.completions.create(
        model = GPT_MODEL, 
        messages = prompt
        )

    in_rate = IN_RATE
    out_rate = OUT_RATE

    tokens_input = prompt_gpt.usage.prompt_tokens
    tokens_output = prompt_gpt.usage.completion_tokens

    tokens_in_column.append(tokens_input)
    tokens_out_column.append(tokens_output)

    if total_input_cost is None:
        total_input_cost = tokens_input / 1_000_000 * in_rate
        total_output_cost = tokens_output / 1_000_000 * out_rate
    else:
        total_input_cost = total_input_cost + (tokens_input / 1_000_000 * in_rate)
        total_output_cost = total_output_cost + (tokens_output / 1_000_000 * out_rate)

    response = prompt_gpt.choices[0].message.content

    cleaned = response.strip()
    if cleaned == "1":
        response_strip = 1
    elif cleaned == "0":
        response_strip = 0
    else:
        response_strip = None
    print(f"Unexpected response at row {row}: {response!r}")

    code_gpt.append(response)

    if df.loc[row, "code_human"] == 1 and response_strip == 1:
        tp = tp + 1
    elif df.loc[row, "code_human"] == 0 and response_strip == 0:
        tn = tn + 1
    elif df.loc[row, "code_human"] == 0 and response_strip == 1:
        fp = fp + 1
    elif df.loc[row, "code_human"] == 1 and response_strip == 0:
        fn = fn + 1


# end runtime counter
end = time.perf_counter()

df[f"code_{GPT_MODEL}"] = code_gpt

# ---- save the results ----
# classifications
results_data_file = f"{data_filename}_{GPT_MODEL}.xlsx"
path_to_data_results = RESULTS_DIR / "nonlocal" / results_data_file

df.to_excel(path_to_data_results, index = False)


# model performance
path_to_model_results = RESULTS_DIR / "classification.xlsx"
results = pd.read_excel(path_to_model_results, sheet_name = f"{DATA_SOURCE}")

new_row = {"model": GPT_MODEL,
           "utterances": df.shape[0],
           "tp": tp,
           "tn": tn,
           "fp": fp,
           "fn": fn,
           "cost": total_input_cost + total_output_cost,
           "runtime": end - start
           }

results = pd.concat([results, pd.DataFrame([new_row])], ignore_index = True)

with pd.ExcelWriter(
    path_to_model_results, engine = "openpyxl", mode = "a", if_sheet_exists = "replace") as writer:
    results.to_excel(writer, sheet_name=f"{DATA_SOURCE}", index = False)


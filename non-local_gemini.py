import os
import time
import random

from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel
from enum import Enum
from tqdm import tqdm

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

# format output
class Answer(str, Enum):
    yes = "yes"
    no = "no"

class Classification(BaseModel):
    explanation: str  # comes first, so reasoning happens before the label
    label: Answer

# ---- get data ----
data_filename = f"cgi_{DATA_SOURCE}"
path_to_data = AIMECON_DATA_DIR / f"{data_filename}.xlsx"
df = pd.read_excel(path_to_data)
df = df.sample(n = 5, ignore_index = True)

# ---- set model params ----
MODEL = "gemini-3.6-flash" # https://ai.google.dev/gemini-api/docs/models

IN_RATE = 2.00
OUT_RATE = 12.00

# ---- prompt GPT ----
# get api key: https://platform.openai.com/api-keys

# check that the api key got loaded in the .env
# if loaded, prints the key
# if not loaded, prints ERROR
key = "GOOGLE_API_KEY"
print(os.environ.get(key, f"ERROR: Variable {key} Not Found"))

# initialize model
client = genai.Client()

# account for high demand
def call_with_retry(client, model, contents, config, max_retries=5):
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(
                model = model,
                contents = contents,
                config = config,
            )
        except Exception as e:
            if "503" in str(e) or "UNAVAILABLE" in str(e):
                wait = (2 ** attempt) + random.uniform(0, 1)
                print(f"503 error, retrying in {wait:.1f}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Failed after {max_retries} retries")

# initialize output objects
tokens_in_column = []
tokens_out_column = []

total_input_cost = None
total_output_cost = None

code_gpt = []
explanation_gpt = []

tp = 0
tn = 0
fp = 0
fn = 0

start = time.perf_counter() # start runtime counter

# FOR TESTING
row = 0
for row in tqdm(df.index):
    CASE = df.loc[row, "text"]

    # construct prompt w case
    PROMPT = (prompts.loc[prompts.id == "Coding", "prompt"].item() + 
              prompts.loc[prompts.id == "Construct", "prompt"].item() +
              prompts.loc[prompts.id == "Prompt1", "prompt"].item() +
              f"\"\"\"{CASE}\"\"\"" + 
              prompts.loc[prompts.id == "Format", "prompt"].item()
              )

    # format prompt for gpt
    prompt = [
        #{"role": "system", "content": CONTEXT},
        {"role": "user", "content": PROMPT}
        ]

    # model settings
    # https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
    try:
        response = response = call_with_retry(
            client,
            MODEL,
            PROMPT,
            types.GenerateContentConfig(
                response_mime_type = "application/json",
                response_schema = Classification,
                automatic_function_calling = types.AutomaticFunctionCallingConfig(disable = True),
            ),
        )
        parsed = Classification.model_validate_json(response.text)
    except Exception as e:
        print(f"Row {row} failed: {e}")
        parsed = None
        response = None

    if parsed is None:
        label = None
        tokens_input = 0
        tokens_output = 0
    else:
        label = parsed.label
        tokens_input = response.usage_metadata.prompt_token_count
        tokens_output = response.usage_metadata.candidates_token_count

    tokens_in_column.append(tokens_input)
    tokens_out_column.append(tokens_output)

    if total_input_cost is None:
        total_input_cost = tokens_input / 1_000_000 * IN_RATE
        total_output_cost = tokens_output / 1_000_000 * OUT_RATE
    else:
        total_input_cost = total_input_cost + (tokens_input / 1_000_000 * IN_RATE)
        total_output_cost = total_output_cost + (tokens_output / 1_000_000 * OUT_RATE)

    if label == "yes":
        response_code = 1
    elif label == "no":
        response_code = 0
    else:
        response_code = None
        print(f"Unexpected response at row {row}: {label!r}")

    code_gpt.append(response_code)
    explanation_gpt.append(parsed.explanation if parsed else None)

    if df.loc[row, "code_human"] == 1 and response_code == 1:
        tp = tp + 1
    elif df.loc[row, "code_human"] == 0 and response_code == 0:
        tn = tn + 1
    elif df.loc[row, "code_human"] == 0 and response_code == 1:
        fp = fp + 1
    elif df.loc[row, "code_human"] == 1 and response_code == 0:
        fn = fn + 1


# end runtime counter
end = time.perf_counter()

df[f"code_{MODEL}"] = code_gpt
df[f"explanation_{MODEL}"] = explanation_gpt

# ---- save the results ----
# classifications
results_data_file = f"{data_filename}_{MODEL}.xlsx"
path_to_data_results = RESULTS_DIR / "nonlocal" / results_data_file

df.to_excel(path_to_data_results, index = False)


# model performance
path_to_model_results = RESULTS_DIR / "classification.xlsx"
results = pd.read_excel(path_to_model_results, sheet_name = f"{DATA_SOURCE}")

new_row = {"model": MODEL,
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
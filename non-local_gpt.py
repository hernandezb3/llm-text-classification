import os

from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from google import genai
from google.genai import types
import anthropic

load_dotenv()


FOCUS_DIR = Path("/Users/brittneyhernandez/Library/CloudStorage/OneDrive-UniversityofConnecticut/focus/project focus/focus_main")
AIMECON_DIR = Path("/Users/brittneyhernandez/Library/CloudStorage/OneDrive-UniversityofConnecticut/AIME-con")

FOCUS_DATA_DIR = FOCUS_DIR / "data/prompt_codes/cgi"
AIMECON_DATA_DIR = AIMECON_DIR / "data"

TRAIN_FILE = AIMECON_DATA_DIR / "cgi_train.xlsx"
VAL_FILE = AIMECON_DATA_DIR / "cgi_validate.xlsx"
TEST_FILE = AIMECON_DATA_DIR / "cgi_test.xlsx"

# ---- get prompt codebook ----
path_to_prompts = AIMECON_DIR / "data_management" / "prompt_codebook.xlsx"
prompts = pd.read_excel(path_to_prompts)

# ---- get data ----
path_to_data = TRAIN_FILE
df = pd.read_excel(path_to_data)
df_sample = df.sample(n = 30, ignore_index = True)

CASE = df.loc[134, "text"]

PROMPT = (prompts.loc[prompts.id == "Coding", "prompt"].item() + 
          prompts.loc[prompts.id == "Construct", "prompt"].item() +
          prompts.loc[prompts.id == "Prompt1", "prompt"].item() +
          f"\"{CASE}\"" + 
          prompts.loc[prompts.id == "Format", "prompt"].item()
          )



GPT_MODEL = "gpt-5.6-terra" # https://developers.openai.com/api/docs/models/all
IN_RATE = 2.00
OUT_RATE = 12.00

CONTEXT = "You are an educational researcher" # SYSTEM PROMPT
TOKENS = 100
TEMPERATURE = 0.2


# ---- GPT ----
# get api key: https://platform.openai.com/api-keys

# check that the api key got loaded in the .env
# if loaded, prints the key
# if not loaded, prints ERROR
key = "OPENAI_API_KEY"
print(os.environ.get(key, f"ERROR: Variable {key} Not Found"))

# initialize model
openai = OpenAI()

# format prompt
prompt = [
    #{"role": "system", "content": CONTEXT},
    {"role": "user", "content": PROMPT}
  ]

# model settings
# https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
prompt_gpt = openai.chat.completions.create(
    model = GPT_MODEL, 
    messages = prompt,
    temperature = TEMPERATURE # range = 0-2
    )

response = prompt_gpt.choices[0].message.content
# https://developers.openai.com/api/docs/models/gpt-4o
in_rate = 2.50
out_rate = 10.00
input_cost = prompt_gpt.usage.prompt_tokens / 1_000_000 * in_rate
output_cost = prompt_gpt.usage.completion_tokens / 1_000_000 * out_rate

print(f"{GPT_MODEL} classified this as a {response}")
print(f"Prompt Tokens = {prompt_gpt.usage.prompt_tokens} (${input_cost})")
print(f"Completion Tokens = {prompt_gpt.usage.completion_tokens} (${output_cost})")





total_input_cost = None
total_output_cost = None
code_gpt = []

for row in df.index:
    CASE = df.loc[row, "text"]

    PROMPT = (prompts.loc[prompts.id == "Coding", "prompt"].item() + 
          prompts.loc[prompts.id == "Construct", "prompt"].item() +
          prompts.loc[prompts.id == "Prompt1", "prompt"].item() +
          f"\"{CASE}\"" + 
          prompts.loc[prompts.id == "Format", "prompt"].item()
          )
    # format prompt
    prompt = [
        #{"role": "system", "content": CONTEXT},
        {"role": "user", "content": PROMPT}
        ]

    # model settings
    # https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
    prompt_gpt = openai.chat.completions.create(
        model = GPT_MODEL, 
        messages = prompt,
        temperature = TEMPERATURE # range = 0-2
        )

    in_rate = IN_RATE
    out_rate = OUT_RATE

    if total_input_cost is None:
        total_input_cost = prompt_gpt.usage.prompt_tokens / 1_000_000 * in_rate
        total_output_cost = prompt_gpt.usage.completion_tokens / 1_000_000 * out_rate
    else:
        total_input_cost = total_input_cost + (prompt_gpt.usage.prompt_tokens / 1_000_000 * in_rate)
        total_output_cost = total_output_cost + (prompt_gpt.usage.completion_tokens / 1_000_000 * out_rate)

















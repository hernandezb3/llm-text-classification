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
df_sample = df.sample(n = 10, random_state = 42, ignore_index = True)

# cases used in ppt examples (df row will be csv row - 2 -- index and column headings)
# 136 = prompt
# 207 = prompt
# 223 = not a prompt
# 3558 = not a prompt
DATA = "Okay, and what would you do?"

PROMPT = f"Classify the following utterance as either a 1 = dialogic prompt or 0 = not a dialogic prompt. A dialogic prompt is defined as an utterance that implies, encourages, requests, or expects a new speaker (or multiple new speakers) to make a verbal contribution. Here is the utterance: \"{DATA}\" Return only 0 or 1."


DATA = df.loc[134, "text"]

PROMPT = (prompts.loc[prompts.id == "Coding", "prompt"].item() + 
          prompts.loc[prompts.id == "Construct", "prompt"].item() +
          prompts.loc[prompts.id == "Prompt1", "prompt"].item() +
          f"\"{DATA}\"" + 
          prompts.loc[prompts.id == "Format", "prompt"].item()
          )


CONTEXT = "You are an educational researcher" # SYSTEM PROMPT

TOKENS = 100
TEMPERATURE = 0.2

GPT_MODEL = "gpt-4o-2024-08-06" # https://developers.openai.com/api/docs/models/all
CLAUDE_MODEL = "claude-3-5-sonnet-20240620" # https://platform.claude.com/docs/en/about-claude/models/overview
GEMINI_MODEL = "gemini-3.6-flash" # https://ai.google.dev/gemini-api/docs/models


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
    #temperature = TEMPERATURE # range = 0-2
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



















# ---- Claude ----
# get api key: https://platform.claude.com/dashboard

# check that the api key got loaded in the .env
# if loaded, prints the key
# if not loaded, prints ERROR
key = "ANTHROPIC_API_KEY"
print(os.environ.get(key, f"ERROR: Variable {key} Not Found"))

# initialize model
claude = anthropic.Anthropic()

# model settings
# https://platform.claude.com/docs/en/api/messages/create
prompt_claude = claude.messages.create(
    model = CLAUDE_MODEL,
    max_tokens = TOKENS,
    temperature = TEMPERATURE, # range = 0-1
    system = CONTEXT,
    messages = [{"role": "user", "content": PROMPT},],
    )

response = prompt_claude.content[0].text

print(response)



# ---- Gemini ---- 
# get api key: https://aistudio.google.com/

# check that the api key got loaded in the .env
# if loaded, prints the key
# if not loaded, prints ERROR
key = "GOOGLE_API_KEY"
print(os.environ.get(key, f"ERROR: Variable {key} Not Found"))

# initialize model
client = genai.Client()

# model settings
# https://ai.google.dev/api/generate-content
prompt_gemini = client.models.generate_content(
    model = GEMINI_MODEL,
    contents = types.Part.from_text(text = PROMPT),
    config = types.GenerateContentConfig(
        temperature = TEMPERATURE, # range = 0-2
        ),
    )
# prompt
response = prompt_gemini.text

print(response)










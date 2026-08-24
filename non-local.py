import os

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from google import genai
from google.genai import types
import anthropic

load_dotenv()

USER_PATH = os.getenv("USER_PATH")
GITHUB_PATH = USER_PATH + "Documents/github"
ONEDRIVE_PATH = USER_PATH + "Library/CloudStorage/OneDrive-UniversityofConnecticut/"
AIMECON_PATH = ONEDRIVE_PATH + "AIME-con/"
FOCUS_PATH = ONEDRIVE_PATH + "focus/project focus/focus_main/"


# ---- get prompt codebook ----
path_to_prompts = AIMECON_PATH + "data_management/"
prompts = pd.read_excel(path_to_prompts + "prompt_codebook.xlsx")

PROMPT = prompts.loc[prompts.id == "Coding", "prompt"].item() + prompts.loc[prompts.id == "Prompt1", "prompt"].item()

# ---- get data ----
path_to_data = FOCUS_PATH + "data/prompt_codes/cgi/clean/"
df = pd.read_excel(path_to_data + "cgi_full_data.xlsx")
df_sample = df.sample(n = 10, random_state = 42, ignore_index = True)
DATA = df.loc[0, "Text"]

prompt_case = PROMPT + " " + DATA
CONTEXT = "You are an educational researcher" # SYSTEM PROMPT

TOKENS = 100
TEMPERATURE = 0.2

GPT_MODEL = "o3-2025-04-16" # https://developers.openai.com/api/docs/models/all
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
    {"role": "system", "content": CONTEXT},
    {"role": "user", "content": prompt_case}
  ]

# model settings
# https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
prompt_gpt = openai.chat.completions.create(
    model = GPT_MODEL, 
    messages = prompt,
    temperature = TEMPERATURE # range = 0-2
    )

response = prompt_gpt.choices[0].message.content

print(response)



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
    messages = [{"role": "user", "content": prompt_case},],
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
    contents = types.Part.from_text(text = prompt_case),
    config = types.GenerateContentConfig(
        temperature = TEMPERATURE, # range = 0-2
        ),
    )
# prompt
response = prompt_gemini.text

print(response)









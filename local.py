import os
import re

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

PROMPT = (prompts.loc[prompts.id == "Coding", "prompt"].item() + " " +
          prompts.loc[prompts.id == "Construct", "prompt"].item() + " " +
          prompts.loc[prompts.id == "Prompt1", "prompt"].item())

output_structure = "Format your response as a number surrounded by three ticks. For example, ```1```"


# ---- get data ----
path_to_data = FOCUS_PATH + "data/prompt_codes/cgi/clean/"
df = pd.read_excel(path_to_data + "cgi_full_data.xlsx")
df.columns = map(str.lower, df.columns) # columns to lowercase
df = df[df["text"].isna() == False] # drop rows where the text is empty (load error?)

df_sample = df.sample(n = 10, random_state = 42, ignore_index = True)
DATA = df_sample


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


# FOR TESTING
each_case = 0

classifications = []
for each_case in range(0, len(DATA)):
    CASE = DATA.loc[each_case, "text"]
    prompt_case = PROMPT + " \"" + CASE + "\" " + output_structure
    input = [{"role": "user", "content": prompt_case}]
    output = pipeline(input, max_new_tokens = TOKENS)
    response = output[0]["generated_text"][-1]["content"]
    pattern = r"```(0|1)```"
    structured_response = re.search(pattern, response)
    if structured_response:
        classify_case = int(structured_response.group(1))
    else:
        print(f"no regex pattern detected in response: {response}")
    classifications.append(classify_case)
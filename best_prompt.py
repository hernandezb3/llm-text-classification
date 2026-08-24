import os
import random

from dotenv import load_dotenv
from huggingface_hub import login
import transformers
import torch
import openpyxl
import pandas as pd
import sklearn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm

load_dotenv()

USER_PATH = os.getenv("USER_PATH")
GITHUB_PATH = USER_PATH + "Documents/github"
ONEDRIVE_PATH = USER_PATH + "Library/CloudStorage/OneDrive-UniversityofConnecticut/"
AIMECON_PATH = ONEDRIVE_PATH + "AIME-con/"
FOCUS_PATH = ONEDRIVE_PATH + "focus/project focus/focus_main/"

# ---- get data ----
path_to_cgi_data = FOCUS_PATH + "data/prompt_codes/cgi/clean/"
cgi_df = pd.read_excel(path_to_cgi_data + "cgi_full_data.xlsx")

path_to_focus_data = FOCUS_PATH + "data/clean/phase1-1/combined_transcripts/"
focus_df = ""


# combine + format data
df = cgi_df 

df.columns = map(str.lower, df.columns) # columns to lowercase
df = df[df["text"].isna() == False] # drop rows where the text is empty (load error?)
df = df.reset_index(drop = True) # reset the index 

DATA = df.loc[0, "text"]

# split data train:val:test  
# Zhang et al., 2024 70:20:10 
# Hu et al., 2024 60:15:25 
# Anglin et al., 2026 33:33:33
# Fine-tuning for AIME-Con: 70:15:15
# cross validate Eertink, et al, 2022 similar performance, more certainty with cv
X = df[["transcript", "speaker", "timestamp", "text"]]
y = df[["code_human"]]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size = 0.30, random_state = 42, stratify = y
    )

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size = 0.50, random_state = 42, stratify = y_temp
    )

print(f"Train : {len(X_train):,}  ({len(X_train)/df.shape[0]*100:.1f}%)")
print(f"Val   : {len(X_val):,}   ({len(X_val)/len(df.shape[0])*100:.1f}%)")
print(f"Test  : {len(X_test):,}   ({len(X_test)/len(df.shape[0])*100:.1f}%)")
print("\nClass distribution (stratification check):")





# ---- get prompt codebook ----
path_to_prompts = AIMECON_PATH + "data_management/"


#"construct_definition_p1": 1,
#"construct_definition_verb": 5,
#"construct_definition_p2": 1,

# dictionary where key:value pairs are sheet name : max number of variants that can be selected
# assumes the variants in the sheets are in a column called variant

construct_variants = {"construct_name": 1,
                      "context_description": 1,
                      "task_description": 1,
                      "construct_definition": 1,
                      "criteria": 20,
                      "each_transcript": 1}

full_path = path_to_prompts + "prompt_codebook.xlsx"
n_prompts = 50

# FOR TESTING:
each_variant = list((construct_variants.keys()))[0]

def create_prompt_variants(construct_variants, path, n_prompts):
    all_prompts = {}
    for each in range(0, n_prompts):
        prompt_combo = {}
        # baseline
        for each_variant in list((construct_variants.keys())):
            df = pd.read_excel(path, sheet_name = each_variant)
            prompt_combo[each_variant] = df
            print(f"there were {construct_variants[each_variant]}")
        return prompt_combo











prompt_conditions = pd.read_excel(path_to_prompts + "combinatorial_prompting_conditions.xlsx")

data = df
each_row = 0
each_transcript = list(dict.fromkeys(data["transcript"]))[0]

def construct_prompts(data, prompt_codebook = "", prompt_conditions = ""):
    transcript = ""
    data["full_transcript"] = ""
    unique_transcripts = list(dict.fromkeys(data["transcript"]))

    for each_transcript in unique_transcripts:
        sample = data[data["transcript"] == each_transcript]
        sample["concatenated"] = sample["speaker"] + " " + sample["timestamp"].astype(str) + " " + sample["text"]

        for each_row in sample.index:
            transcript = transcript + sample.loc[each_row, "concatenated"] + " \n"

        #
        data.loc[data["transcript"] == each_transcript, "full_transcript"] = transcript
        text = data.loc[each_row, "text"]
    return data

construct_prompts(df)

PROMPT = prompts.loc[prompts.id == "Code", "prompt"].item() + prompts.loc[prompts.id == "Prompt1", "prompt"].item()







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

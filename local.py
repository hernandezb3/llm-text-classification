import os
import re

from dotenv import load_dotenv
from huggingface_hub import login
import transformers
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

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

output_structure = "Format your response as a 0 or 1 surrounded by three ticks. For example, ```1```"


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
        classify_case = np.nan
    classifications.append(classify_case)

DATA["code_llm"] = classifications

# ---- evaluate performance ----

codes = pd.DataFrame({"y_true": DATA["code_human"], 
                      "y_pred": DATA["code_llm"]})

missing_true = codes[codes["y_true"].isna()]
missing_pred = codes[codes["y_pred"].isna()]

if not missing_true.empty: 
    print(f"\nThere were {missing_true.shape[0]} missing from code_human. The following cases were dropped from evaluation metric calculations: \n {missing_true}\n ")
if not missing_pred.empty: 
    print(f"\nThere were {missing_pred.shape[0]} missing from code_human. The following cases were dropped from evaluation metric calculations. \n {missing_pred}\n ")

codes = codes[codes["y_true"].notna() & codes["y_pred"].notna()]

confusion = confusion_matrix(codes["y_true"], codes["y_pred"])
print(f"\nCONFUSION MATRIX\n{confusion}")

TN, FP, FN, TP = confusion.ravel()

# True negative rate or specificity: proporition of correctly id'd negatives of all that are actually negative
TNR = TN/(TN+FP) 
# Negative predictive value: proportion of negatives that are false
NPV = TN/(TN+FN)
# False positive rate or fall out: proportion of incorrectly id'd negatives 
FPR = FP/(TN+FP)
# False negative rate
FNR = FN/(TP+FN)
# False discovery rate: proportion of positives that are false
FDR = FP/(TP+FP)

# Overall accuracy
accuracy = (TP+TN)/(TP+FP+FN+TN)
# Precision or positive predictive value
precision = TP/(TP+FP)
# Sensitivity, hit rate, recall, or true positive rate
recall = TP/(TP+FN)
# F1 score
f1 = (2*TP)/(2*TP + FP + FN)

print(f"""\nEVALUATION METRICS \n
n = {codes.shape[0]}
n Dropped = {DATA.shape[0] - codes.shape[0]}
Accuracy = {accuracy}
Precision = {precision}
Recall = {recall}
F1 = {f1}
True Negative Rate = {TNR}
Negative Predictive Value = {NPV}
False Positive Rate = {FPR}
False Negative Rate = {FNR}
False Discovery Rate = {FDR}
""")


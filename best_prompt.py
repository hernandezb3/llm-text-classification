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


# FILE STRUCTURE FOR HPC/COLAB
# .env in cd
# data to data/
# prompt_codebook to data_management/
# classifications.xlsx to results/
# make sure results/local exists

load_dotenv()

USER = "hpc" 

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
DATA_SOURCE = "train" # train, validate, test
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\nUsing device: {DEVICE}")
if DEVICE == "cuda":
    print(torch.cuda.get_device_name(0))

# ---- get data ----
data_filename = f"cgi_{DATA_SOURCE}"
path_to_data = DATA_DIR / f"{data_filename}.xlsx"
df = pd.read_excel(path_to_data)
#df = df.sample(n = 5, ignore_index = True)
# call out in the room, what performance did you estimate
# performance metrics are estimates > seguey to uncertainty




# ---- get prompt codebook ----
path_to_prompts = WORKING_DIR / "data_management" / "prompt_codebook.xlsx"
prompts = pd.read_excel(path_to_prompts)
N_VARIANTS = 50

#"construct_definition_p1": 1,
#"construct_definition_verb": 5,
#"construct_definition_p2": 1,

# dictionary where key:value pairs are sheet name : max number of variants that can be selected
# assumes the variants in the sheets are in a column called variant

construct_variants = {"construct_name": 1,
                      "context_description": 1,
                      "task_description": 1,
                      "construct_definition": 1,
                      "criteria": 20}




# FOR TESTING:
each_variant = list((construct_variants.keys()))[0]

def create_prompt_variants(construct_variants, path, n_prompts):
    all_prompts = {}
    for each in range(0, N_VARIANTS):
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

MODEL = "ibm-granite/granite-4.2-8b"
TASK = "text-generation"
TOKENS = 500
TEMPERATURE = 0.1
QUANTIZATION = torch.bfloat16 # can use bfloat16 or bfloat32 if cuda is available (float for cpu, bfloat for gpu)

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
#                               model_kwargs = {"dtype": QUANTIZATION}, 
#                               token = os.getenv("HF_TOKEN")) # initate pipeline

model = AutoModelForCausalLM.from_pretrained(MODEL, 
                                              dtype = QUANTIZATION,
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

    PROMPT = (prompts.loc[prompts.id == "Coding2", "prompt"].item() + " " +
                  prompts.loc[prompts.id == "Construct", "prompt"].item() + " " +
                  prompts.loc[prompts.id == "Prompt1", "prompt"].item() + " " +
                  f"\n\"\"\"{CASE}\"\"\"\n" + " " +
                  prompts.loc[prompts.id == "Format2", "prompt"].item()
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
if "/" in MODEL:
    model_stripped = re.split("/", MODEL)[1]
else:
    model_stripped = MODEL

results_data_file = f"{data_filename}_{model_stripped}.xlsx"
path_to_data_results = RESULTS_DIR / "local" / results_data_file


# model performance
path_to_model_results = RESULTS_DIR / "classification.xlsx"
results = pd.read_excel(path_to_model_results, sheet_name = f"{DATA_SOURCE}")

new_row = {"model": MODEL,
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



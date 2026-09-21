import random
import re

from dotenv import load_dotenv
from pathlib import Path
import pandas as pd

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
    drive.mount("/content/drive/")
    WORKING_DIR = Path.cwd()
    DATA_DIR = WORKING_DIR / "data"

DATA_MANAGEMENT_DIR = WORKING_DIR / "data_management"
RESULTS_DIR = WORKING_DIR / "results"

# ---- get prompt codebook ----
path_to_prompts = DATA_MANAGEMENT_DIR / "empirical_prompt_variants.xlsx"

prompt_index = 0
number_of_prompts = 50
all_prompt_ids = []

prompt_manifest = []

# FOR TESTING
i = 0
for i in range(0, number_of_prompts):
    prompt_dictionary = {}
    pattern = r"[A-Za-z]+_(\d+)"

    # randomly select 1 context variant
    context_sheet = pd.read_excel(path_to_prompts, sheet_name = "context")
    max_context = max(context_sheet.index)
    n_context = 1
    context_row = random.randint(0, max_context)
    prompt_dictionary["context"] = context_sheet.loc[context_row, "variant"]
    context_id = re.search(pattern, context_sheet.loc[context_row, "id"]).group(1)

    # randomly select 1 task variant
    task_sheet = pd.read_excel(path_to_prompts, sheet_name = "task")
    max_task = max(task_sheet.index)
    n_task = 1
    task_row = random.randint(0, max_task)
    prompt_dictionary["task"] = task_sheet.loc[task_row, "variant"]
    task_id = re.search(pattern, task_sheet.loc[task_row, "id"]).group(1)

    # randomly select one construct definition variant
    definition_sheet = pd.read_excel(path_to_prompts, sheet_name = "definition")
    max_definition = max(definition_sheet.index)
    n_definition = 1
    definition_row = random.randint(0, max_definition)
    prompt_dictionary["definition"] = definition_sheet.loc[definition_row, "variant"]
    definition_id = re.search(pattern, definition_sheet.loc[definition_row, "id"]).group(1)

    # randomly draw 0-n guidance
    guidance_sheet = pd.read_excel(path_to_prompts, sheet_name = "guidance")
    total_guidance = guidance_sheet.shape[0]
    n_guidance = random.randint(0, total_guidance)
    guidance_rows = random.sample(range(0, total_guidance), n_guidance)
    guidance_rows.sort()

    guidance_ids = []
    guidance = ""

    # FOR TESTING
    row = 1
    for row in guidance_rows:
        guidance_ids.append(guidance_sheet.loc[row, "id"])
        guidance_row = guidance_sheet.loc[row, "variant"]
        guidance = guidance + guidance_row + "\n"

    prompt_dictionary["n_guidance"] = n_guidance
    prompt_dictionary["guidance_ids"] = guidance_ids
    prompt_dictionary["guidance"] = guidance

    format_sheet = pd.read_excel(path_to_prompts, sheet_name = "format")
    prompt_dictionary["format_local"] = format_sheet.loc[format_sheet.id == "format_local", "variant"].item()
    prompt_dictionary["format_case"] = format_sheet.loc[format_sheet.id == "format_case", "variant"].item()
    prompt_dictionary["format_guidance"] = format_sheet.loc[format_sheet.id == "format_guidance", "variant"].item()

    prompt_id = f"c{context_id}t{task_id}d{definition_id}g{n_guidance}"

    # we want 50 unique prompts so
    # if the id is a duplicate of a past id, go back and try again
    if prompt_id in all_prompt_ids:
        i =- 1
        continue

    all_prompt_ids.append(prompt_id)
    full_prompt_id = f"prompt{i}_{prompt_id}"
    prompt_index =+ 1

    prompt_dictionary["prompt_id"] = full_prompt_id

    prompt_manifest.append(prompt_dictionary)

df = pd.DataFrame(prompt_manifest)
df = df[["prompt_id", "guidance_ids", "context", "task", "definition", "format_guidance", "guidance", "format_case", "format_local"]]

df.to_csv(DATA_MANAGEMENT_DIR / f"empirical_prompts_{number_of_prompts}.csv", index=False, encoding="utf-8-sig")
df.to_pickle(DATA_MANAGEMENT_DIR / f"empirical_prompts_{number_of_prompts}.pkl")

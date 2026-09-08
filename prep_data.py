import random

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

FOCUS_DATA_DIR = Path("/Users/brittneyhernandez/Library/CloudStorage/OneDrive-UniversityofConnecticut/focus/project focus/focus_main/data/prompt_codes/cgi")
AIMECON_DATA_DIR = Path("/Users/brittneyhernandez/Library/CloudStorage/OneDrive-UniversityofConnecticut/AIME-con/data")

CODING_MANIFEST = FOCUS_DATA_DIR / "coding_assignments_all.xlsx"

SOURCE_FILE = AIMECON_DATA_DIR / "cgi_finetune_data.xlsx"
TRAIN_FILE = AIMECON_DATA_DIR / "cgi_train.xlsx"
VAL_FILE = AIMECON_DATA_DIR / "cgi_validate.xlsx"
TEST_FILE = AIMECON_DATA_DIR / "cgi_test.xlsx"

kyli_cgi = pd.read_parquet("/Users/brittneyhernandez/Library/CloudStorage/OneDrive-UniversityofConnecticut/kylies replication/data_cgi/cgi_train.parquet")

manifest = pd.read_excel(CODING_MANIFEST)
df = pd.read_excel(SOURCE_FILE)

# split data 33:33:33
labels = {1: "train",
          2: "validate",
          3: "test"}

n_iterations = manifest.shape[0]

for row in range(n_iterations):
    value = random.randint(1, 3) 
    manifest.loc[row, "split"] = value

manifest['split'] = manifest['split'].map(labels)
manifest.to_excel(FOCUS_DATA_DIR / "coding_assignments_metadata.xlsx", index = False)
print(f"Updated file metadata.")

training_files = manifest["file"][manifest["split"] == "train"]
validation_files = manifest["file"][manifest["split"] == "validate"]
testing_files = manifest["file"][manifest["split"] == "test"]

training = df[df['filename'].isin(training_files)]
validation = df[df['filename'].isin(validation_files)]
testing = df[df['filename'].isin(testing_files)]

# FOR TESTING
each_df = training
for each_df in [training, validation, testing]:
    files = len(set(each_df["filename"]))
    utterances = each_df.shape[0]
    prompts = each_df[each_df["code_human"] == 1].shape[0]
    print(f"{each_df} df has:\nfiles = {files},\nutterance = {utterances},\ndialogic prompts = {prompts}")

training.to_csv(TRAIN_FILE)
validation.to_csv(VAL_FILE)
testing.to_csv(TEST_FILE)
    
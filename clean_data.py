import re
from itertools import combinations

from pathlib import Path
import pandas as pd
from sklearn.metrics import cohen_kappa_score


FOCUS_DATA_DIR = Path("/Users/brittneyhernandez/Library/CloudStorage/OneDrive-UniversityofConnecticut/focus/project focus/focus_main/data/prompt_codes/cgi")
AIMECON_DATA_DIR = Path("/Users/brittneyhernandez/Library/CloudStorage/OneDrive-UniversityofConnecticut/AIME-con/data")

CODING_MANIFEST = FOCUS_DATA_DIR / "coding_assignments_all.xlsx"

SOURCE_FILE = AIMECON_DATA_DIR / "cgi_finetune_data.xlsx"
TRAIN_FILE = AIMECON_DATA_DIR / "cgi_train.parquet"
DEV_FILE = AIMECON_DATA_DIR / "cgi_dev.parquet"
TEST_FILE = AIMECON_DATA_DIR / "cgi_test.parquet"


MODEL_NAME = "Qwen/Qwen2-7B-Instruct"
MAX_INPUT_TOKENS = 2048

CODERS = ["hima", "kelsi", "brittney"]

manifest = pd.read_excel(CODING_MANIFEST)
manifest["sample_size"] = None

# FOR TESTING
row = 3
item = 0
each_pair = ('prompt_hima', 'prompt_kelsi')

all_merged_files = None
for row in range(0, manifest.shape[0]):
    file = manifest.loc[row, "file"]
    rows_coded = manifest.loc[row, "total_turns"]
    start = manifest.loc[row, "row_to_start"]
    end = start + 100

    pattern = r"([0-9_A-Za-z]*)"
    name = re.search(pattern, file).group(0)
    pattern = r"(.xlsx)"
    ext = re.search(pattern, file).group(0)
    file_name = name + "_coding" + ext

    print(f"\n🚦 Starting merge for file: {file_name}\n")

    merged = None
    for item in range(0, len(CODERS)):
        coder_name = CODERS[item]
        file_path = FOCUS_DATA_DIR / f"{coder_name}/{file_name}"
        df = pd.read_excel(file_path)
        df.columns = df.columns.str.lower()

        if end > df.shape[0]:
            end = df.shape[0]
            
        sample = df.iloc[start:end]
        #nas = sample['dialogic_prompt'].isna().sum()
        #sample['dialogic_prompt'] = sample['dialogic_prompt'].fillna(0)

        if merged is None:
            merged = sample
            merged.insert(0, 'index', merged.index)
            merged = merged.drop(columns = ["needs_redacting"])
            merged.rename(columns = {"dialogic_prompt": f"prompt_{coder_name}"}, 
                          inplace = True)
        else:
            merged[f"prompt_{coder_name}"] = sample["dialogic_prompt"]

        print(f"✔️ Added the prompt_{coder_name}")

    cols = merged.columns[merged.columns.str.startswith('prompt_')]
    merged["code_human"] = merged[cols].mode(axis = 1)[0]

    nas = merged[cols].isna().sum()
    merged = merged.dropna(subset = cols)

    print(f"There were {nas} missing cases found.")

    manifest.loc[row, "sample_size"] = merged.shape[0]

    coder_cols = list(merged[cols].columns)
    pairs = list(combinations(coder_cols, 2))

    if merged.shape[0] > 0:
        for each_pair in pairs:
            column_name = f"kappa_{each_pair[0]}_{each_pair[1]}"
            if column_name not in manifest.columns:
                manifest[column_name] = None
            kappa = cohen_kappa_score(merged[each_pair[0]], merged[each_pair[1]])
            manifest.loc[row, column_name] = kappa
            print(f"Kappa for {file_name} {each_pair} = {round(kappa, ndigits = 4)}")

    if all_merged_files is None:
        all_merged_files = merged
    else:
        all_merged_files = pd.concat([all_merged_files, merged], axis = 0)

    print(f"Added {file_name} to all_merged_files")

manifest.to_excel(FOCUS_DATA_DIR / "coding_assignments_kappa.xlsx", index = False)
all_merged_files.to_excel(SOURCE_FILE, index = False)
print(f"Saved cgi_finetune_data.xlsx to {AIMECON_DATA_DIR}")

#manifest[row, "kappa_sample_size"] = merged.shape[0]
    
            

        








#print(f"Prepared data will be saved to {AIMECON_DATA_DIR}")

#raw = pd.read_excel(SOURCE_FILE)

#print(f"Loaded {len(raw):,} rows and {raw.shape[1]} columns")
#print(raw.head(3))
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

CODERS = ["hima", "kelsi", "brittney"]

manifest = pd.read_excel(CODING_MANIFEST)

# FOR TESTING
row = 5
item = 1
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

    print(f"\n🚦 ({row}) Starting merge for file: {file_name}\n")

    merged = None
    for item in range(0, len(CODERS)):
        coder_name = CODERS[item]
        file_path = FOCUS_DATA_DIR / f"{coder_name}/{file_name}"
        df = pd.read_excel(file_path)
        df.columns = df.columns.str.lower()

        if end > df.shape[0]:
            end = df.shape[0]
            
        sample = df.iloc[start:end]

        if merged is None:
            merged = sample
            merged.insert(0, 'index', merged.index)
            merged.insert(1, "filename", file)
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

    print(f"\nThe number of missing cases dropped were:\n{nas}\n")

    manifest.loc[row, "actual_n"] = merged.shape[0]            
    manifest.loc[row, "target_n"] = end - start
    manifest.loc[row, "actual_target"] = (manifest.loc[row, "actual_n"] == manifest.loc[row, "target_n"])
    manifest.loc[row, "teacher_turns"] = list(merged["speaker"]).count("Teacher")
    manifest.loc[row, "prevelance"] = (merged['code_human'] == 1).sum()
    no_q_prompt = ((merged["text"].str.contains(r"\?") == False) & (merged['code_human'] == 1)).sum()
    q_no_prompt = ((merged["text"].str.contains(r"\?")) & (merged['code_human'] == 0)).sum()
    manifest.loc[row, "tricky"] = no_q_prompt + q_no_prompt

    coder_cols = list(merged[cols].columns)
    pairs = list(combinations(coder_cols, 2))

    if merged.shape[0] > 0:
        for each_pair in pairs:
            coder1 = re.sub("prompt_", "", each_pair[0])
            coder2 = re.sub("prompt_", "", each_pair[1])
            column_name = f"kappa_{coder1}_{coder2}"
            if column_name not in manifest.columns:
                manifest[column_name] = None
            kappa = cohen_kappa_score(merged[each_pair[0]], merged[each_pair[1]])
            manifest.loc[row, column_name] = kappa
            print(f"Kappa for {file_name} {each_pair} = {round(kappa, ndigits = 4)}")

    if all_merged_files is None:
        all_merged_files = merged
    else:
        all_merged_files = pd.concat([all_merged_files, merged], axis = 0)

    print(f"\nAdded {file_name} to all_merged_files\n")

manifest.to_excel(FOCUS_DATA_DIR / "coding_assignments_metadata.xlsx", index = False)
all_merged_files.to_excel(SOURCE_FILE, index = False)
print(f"Saved cgi_finetune_data.xlsx to {AIMECON_DATA_DIR}")

#manifest[row, "kappa_sample_size"] = merged.shape[0]
    
            

        








#print(f"Prepared data will be saved to {AIMECON_DATA_DIR}")

#raw = pd.read_excel(SOURCE_FILE)

#print(f"Loaded {len(raw):,} rows and {raw.shape[1]} columns")
#print(raw.head(3))
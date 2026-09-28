
from itertools import combinations
from dotenv import load_dotenv
from pathlib import Path
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score


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
    drive.mount('/content/drive/')
    WORKING_DIR = Path.cwd()
    DATA_DIR = WORKING_DIR / "data"

RESULTS_DIR = WORKING_DIR / "results"

train = pd.read_excel(DATA_DIR / "cgi_train.xlsx")
dev = pd.read_excel(DATA_DIR / "cgi_dev.xlsx")
test = pd.read_excel(DATA_DIR / "cgi_test.xlsx")
all = pd.read_excel(DATA_DIR / "cgi_all.xlsx")

names = ["coder 1", "coder 2", "coder 3", "vote"]
train_kappa = pd.DataFrame(index = names, columns = names)
dev_kappa = pd.DataFrame(index = names, columns = names)
test_kappa = pd.DataFrame(index = names, columns = names)
all_kappa = pd.DataFrame(index = names, columns = names)

# coder 1,2
cohen_kappa_score(all["prompt_coder1"], all["prompt_coder2"])
cohen_kappa_score(train["prompt_coder1"], train["prompt_coder2"])
cohen_kappa_score(dev["prompt_coder1"], dev["prompt_coder2"])
cohen_kappa_score(test["prompt_coder1"], test["prompt_coder2"])

# coder 1,3
cohen_kappa_score(all["prompt_coder1"], all["prompt_coder3"])
cohen_kappa_score(train["prompt_coder1"], train["prompt_coder3"])
cohen_kappa_score(dev["prompt_coder1"], dev["prompt_coder3"])
cohen_kappa_score(test["prompt_coder1"], test["prompt_coder3"])

# coder 2,3
cohen_kappa_score(all["prompt_coder2"], all["prompt_coder3"])
cohen_kappa_score(train["prompt_coder2"], train["prompt_coder3"])
cohen_kappa_score(dev["prompt_coder2"], dev["prompt_coder3"])
cohen_kappa_score(test["prompt_coder2"], test["prompt_coder3"])

# coder 1, vote
cohen_kappa_score(all["prompt_coder1"], all["code_human"])
cohen_kappa_score(train["prompt_coder1"], train["code_human"])
cohen_kappa_score(dev["prompt_coder1"], dev["code_human"])
cohen_kappa_score(test["prompt_coder1"], test["code_human"])

# coder 2, vote
cohen_kappa_score(all["prompt_coder2"], all["code_human"])
cohen_kappa_score(train["prompt_coder2"], train["code_human"])
cohen_kappa_score(dev["prompt_coder2"], dev["code_human"])
cohen_kappa_score(test["prompt_coder2"], test["code_human"])

# coder 3, vote
cohen_kappa_score(all["prompt_coder3"], all["code_human"])
cohen_kappa_score(train["prompt_coder3"], train["code_human"])
cohen_kappa_score(dev["prompt_coder3"], dev["code_human"])
cohen_kappa_score(test["prompt_coder3"], test["code_human"])

# convert to accuracy??

cols = {
    "coder 1": "prompt_coder1",
    "coder 2": "prompt_coder2",
    "coder 3": "prompt_coder3",
    "vote": "code_human",
    }

names = list(cols)
splits = {"all": all, "train": train, "dev": dev, "test": test}

kappa = {}
for split, data in splits.items():
    k = pd.DataFrame(1.0, index = names, columns = names)  # diagonal = 1
    for a, b in combinations(names, 2):
        k.loc[a, b] = k.loc[b, a] = cohen_kappa_score(data[cols[a]], data[cols[b]])
    kappa[split] = k

print(kappa["train"])
print(kappa["dev"])
print(kappa["test"])
print(kappa["all"])

accuracy = {}
for split, data in splits.items():
    k = pd.DataFrame(1.0, index = names, columns = names)  # diagonal = 1
    for a, b in combinations(names, 2):
        k.loc[a, b] = k.loc[b, a] = accuracy_score(data[cols[a]], data[cols[b]])
    accuracy[split] = k

print(accuracy["train"])
print(accuracy["dev"])
print(accuracy["test"])
print(accuracy["all"])
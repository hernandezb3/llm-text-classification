import re
import ast

from dotenv import load_dotenv
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
DATA_SOURCE = "train" # train, validate, test

path_to_prompts = WORKING_DIR / "data_management" / "empirical_prompts_50.csv"
prompts = pd.read_csv(path_to_prompts)
prompt_ids = prompts["prompt_id"]
prompt_conditions = prompt_ids.str.split("_").str[1]

path_to_performance = WORKING_DIR / "results" / "classification.xlsx"
performance = pd.read_excel(path_to_performance, sheet_name = "train")

# create dataframe
context_vars = [f"c{i}" for i in range(1,7)]
task_vars = [f"t{i}" for i in range(1,6)]
definition_vars = [f"d{i}" for i in range(1,5)]
dummy_vars = context_vars + task_vars + definition_vars

pattern = r"c(\d)t(\d)d(\d)g(\d+)"

components = prompt_conditions.str.extract(pattern)
components.columns = ["c", "t", "d", "g"]

components["id"] = prompt_ids
components["guidance_ids"] = prompts["guidance_ids"]
components = components[["id", "c", "t", "d", "g", "guidance_ids"]]

for i in dummy_vars:
    components[f"{i}"] = 0

row = 0
for row in components.index:
    context_component = f"c{components.loc[row, "c"]}"
    components.loc[row, context_component] = 1

    task_component = f"t{components.loc[row, "t"]}"
    components.loc[row, task_component] = 1

    definition_component = f"d{components.loc[row, "d"]}"
    components.loc[row, definition_component] = 1


def sort_keys(s):
    m = re.match(r"([A-Za-z]+)(\d+)(?:\.(\d+))?", s)
    letters, num, sub = m.groups()
    return (letters, int(num), int(sub) if sub else -1)

prompts["guidance_ids"] = prompts["guidance_ids"].str.replace(" ", "", regex = False)

guidance_components = []
for row in prompts.index:
    id_list = ast.literal_eval(prompts.loc[row, "guidance_ids"])
    if len(id_list) < 1:
        continue
    guidance_components = guidance_components + id_list

guidance_components = sorted(set(guidance_components), key = sort_keys)

for i in guidance_components:
    components[f"{i}"] = 0

for row in components.index:
    id_list = ast.literal_eval(components.loc[row, "guidance_ids"])
    if len(id_list) < 1:
            continue
    for id in id_list:
        components.loc[row, f"{id}"] = 1

components.loc[1, "guidance_ids"]

df = components.merge(performance, left_on = "id", right_on = "prompt")

# ---- prompt component performance ----
def fit_lr(df_x, df_y):
    X = df_x
    y = df_y
    model = LinearRegression()
    fit = model.fit(X, y)

    # residual
    n, k = X.shape

    y_pred = model.predict(X)
    resid = y - y_pred
    sigma_squared = (resid @ resid) / (n-k-1)

    # standard errors
    X_design = np.column_stack([np.ones(n), X])
    cov = sigma_squared * np.linalg.inv(X_design.T @ X_design)
    se = np.sqrt(np.diag(cov))[1:] 

    t = fit.coef_ / se
    p = 2 * stats.t.sf(np.abs(t), df=n - k - 1)

    results = pd.DataFrame({"coef": fit.coef_, "se": se, "t": t, "p": p}, index = X.columns)
    r_squared = r2_score(y, y_pred)

    print(f"\noutcome: {df_y.name}")
    print(f"\nn: {n} \nk: {k} \nr-squared: {r_squared}\n\n")
    print(f"B0: {model.intercept_}")
    print(results)

fit_lr(df[guidance_components], df["accuracy"])
fit_lr(df[guidance_components], df["fpr"])

fit_lr(df[context_vars], df["accuracy"])
fit_lr(df[task_vars], df["accuracy"])
fit_lr(df[definition_vars], df["accuracy"])
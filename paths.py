from pathlib import Path

USER = "local" # either local or colab

if USER == "local":
    WORKING_DIR = Path.cwd()
elif USER == "colab":
    WORKING_DIR = Path('/content/drive/MyDrive/aime-con')

DATA_DIR =  WORKING_DIR / "data" 
CODEBOOK_DIR = WORKING_DIR / "data_management" 
RESULTS_DIR = WORKING_DIR / "results"
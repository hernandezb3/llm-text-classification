from dotenv import load_dotenv
from pathlib import Path

# USER is either colab or local
USER = "local"

if USER == "local":
    WORKING_DIR = Path.cwd()
elif USER == "colab":
    WORKING_DIR = Path('/content/drive/MyDrive/aime-con')

DATA_DIR =  WORKING_DIR / "data" 
CODEBOOK_DIR = WORKING_DIR / "data_management" 
RESULTS_DIR = WORKING_DIR / "results"
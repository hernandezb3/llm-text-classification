from pathlib import Path

USER = "local" # either local or colab

if USER == "local":
    from dotenv import load_dotenv
    WORKING_DIR = Path.cwd()
    load_dotenv()
elif USER == "colab":
    from google.colab import drive
    drive.mount('/content/drive/')
    WORKING_DIR = Path('/content/drive/MyDrive/aime-con')

DATA_DIR =  WORKING_DIR / "data" 
CODEBOOK_DIR = WORKING_DIR / "data_management" 
RESULTS_DIR = WORKING_DIR / "results"
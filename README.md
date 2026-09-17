# Text Classification with Large Language Models
AIMEcon tutorial, *Text Classification with Large Language Models: Pipelines, Fine-tuning, and Measurement Validity*
By Brittney Hernandez, Claudia Ventura, Kylie Anglin

# Prerequisite Knowledge & Skills
Text Classification: 
Conceptual understanding of text classification as a method of analysis, and/or familiarity with traditional classification methods (e.g., bag-of-words, supervised classifiers)

LLM Mechanics: 
Understanding of LLMs as next-token prediction systems, including tokenization, and a broad sense of how training data shapes model behavior.

Measurement Theory: 
Knowledge of Shadish, Cook, & Campbell’s (2002) validity framework.

Programming: 
Proficiency in one or more programming language(s) such as Python or R. Conceptual understanding of file input/output, API calls, data manipulation, functions, and loops. 


# Prerequisite Software & Packages
- HuggingFace Account
- HuggingFace Key
- One of Option A or B

Colab Option A)
- Google account
- Google Colab account


Local Option B)
- VS Code
- Python 3.12.3


# Folder Set-Up
Save the following files. 

Colab Option A)
Save these files to your Google Drive in `My Drive/`. Below is a link to a folder that's already set up in Google Drive.

- [aime-con](https://drive.google.com/drive/folders/1Zd4YEUcThwXW2uRGDFmhKjDPL7zd_JVC?usp=share_link)/


Local Option B)
If you're working locally, clone the [llm-text-classification](https://github.com/hernandezb3/llm-text-classification) GitHub Repo to your computer and download the data files from Google Drive and save them to the `data/` folder in your cloned repo:

- aime-con/[data](https://drive.google.com/drive/folders/1JIynOkgSf21fB5U1FEwm-YLqWHVMnhoH?usp=share_link)/

- aime-con
    - data
        - train.xlsx
        - dev.xlsx
        - test.xlsx
    - data_management
    - results
        - local
        - non-local
        - classifications.txt
    - 00_secrets.txt
    - 00_start.ipynb
    - 01_non-local.ipynb
    - 02_local.ipynb
    - requirements.txt
    - .env*


# Environment

## Kernel: Colab vs Local


## Virtual Environment
We'll use virtual environments to standardize our package repository. 

*Virtual environements* are 


### STEP A: Start a Virtual Environment

To do this:
- Press `CMD + SHIFT + P`
- Select `Python: Create Environment`



check package dependencies
```
pip check
```

check what pip would resolve without actually installing packages
```
pip install --dry-run -r requirements.txt
```

install packages from requirements.txt
assumes the requirements.txt file is in your working directory
```
pip install -r requirements.txt
```

## API Keys


### Save credentials



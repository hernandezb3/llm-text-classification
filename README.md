# Text Classification with Large Language Models
[AIME-Con](https://www.xcdsystem.com/ncme/program/47bbPZ3/index.cfm) tutorial, *Text Classification with Large Language Models: Pipelines, Fine-tuning, and Measurement Validity*
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
- Complete Gated Access Form for Llama 3.2 1B
- One of Option A or B:

Colab Option A)
- Google Drive account
- Google Colab account

Local Option B)
- VS Code
- Python 3.12.3

# Folder Structure
The `aime-con` is set up as follows:

```
|── aime-con/
│   ├── data/
│   │   ├── dev.xlsx
│   │   ├──test.xlsx
│   │   └── train.xlsx
│   ├── data_management/
│   │   ├──human_prompt_codebook.docx
│   │   └── llm_prompt_codebook.xlsx
│   ├── results/
│   │   ├──local/
│   │   ├── non-local/
│   │   ├──prompt-engineering/
│   │   ├── fine-tuning/
│   │   └── classifications.txt
│   ├── README.md
│   │── paths.py
│   │── requirements.txt
│   │── secrets-template.txt
│   │── 00_colab-setup.ipynb
│   │── 00_local-setup.ipynb
│   │── 01_non-local.ipynb
│   │── 02_local.ipynb
│   │── 03_prompt-engineering.ipynb
│   └── 04_fine-tuning.ipynb
```

# Hugging Face
In HuggingFace you will need to sign up for an account, create a read-only acess token, and complete the Community Access Agreement for Llama 3.2. 

Navigate to https://huggingface.co and click Sign Up. 

<p align="center">
<img src="readme_images/hf sign up.png" height="500">
</p>

Click on your profile icon in the top right corner and select Access Tokens. 
<p align="center">
<img src="readme_images/nav to tokens.png" height="500">
</p>

Select Create new Access Token.
<p align="center">
<img src="readme_images/new token.png" height="300">
</p>

Select Read Only, assign a token name and select Create Token. A pop-up window will appear with your access token. Save this somewhere secure (e.g., a password manager). 
<p align="center">
<img src="readme_images/create token.png" height="500">
</p>

Complete the Community Acess Agreement for [Llama 3.2 1B](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) on Hugging Face.

<p align="center">
<img src="readme_images/gated access.png" height="500">
</p>


# Option A: Google Colab

Save the `aime-con` folder and all of it's contents to `My Drive/`. Below is a link to the aime-con folder. *Note.* Do not save it to a folder called Colab Notebooks.

- [aime-con](https://drive.google.com/drive/folders/1Zd4YEUcThwXW2uRGDFmhKjDPL7zd_JVC?usp=share_link)/

Navigate to the file `aime-con/00a_setup-colab.ipynb`, right click on ... and select Open with > Google Colabratory. 

[ADD SCREENSHOT]

Follow the instructions listed in the file, `00a_setup-colab.ipynb`


# Option B: Local

Download VS Code:
Download Python 3.12.3

Clone the [llm-text-classification](https://github.com/hernandezb3/llm-text-classification) GitHub Repo to your computer.

Navigate to where you want the repo saved on your computer.
```
cd path/to/folder/
```

Clone the repo
```
git clone https://github.com/hernandezb3/llm-text-classification.git
```

Download the `train.xlsx`, `dev.xlsx`, and `test.xlsx` files from Google Drive and save them to the `data/` folder in your cloned repo. *Note* There is a .gitignore file in the data file that keeps any data from being pushed to GitHub.

- aime-con/[data](https://drive.google.com/drive/folders/1JIynOkgSf21fB5U1FEwm-YLqWHVMnhoH?usp=share_link)/








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



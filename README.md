# Text Classification with Large Language Models
[AIME-Con](https://www.xcdsystem.com/ncme/program/47bbPZ3/index.cfm) 4-hour training session, *Text Classification with Large Language Models: Pipelines, Fine-tuning, and Measurement Validity*

by Brittney Hernandez, PhD; Claudia Ventura; Kylie Anglin, PhD

**Description** 

This interactive workshop covers methods of binary text classification using large language models: API calls to non-locally hosted models, local models, and finetuned models. Participants will build text-classification pipelines with an emphasis on practicalities and measurement validity. Tradeoffs in approaches (cost, performance, privacy, etc.) will be discussed throughout.

<p align="center">
    <img src="readme_images/focus logo.jpg" height="120">
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="readme_images/uconn-wordmark-stacked-blue.png" height="80">
</p>

# Prerequisite Knowledge & Skills
**Text Classification**

Conceptual understanding of text classification as a method of analysis, and/or familiarity with traditional classification methods (e.g., bag-of-words, supervised classifiers)

**LLM Mechanics**

Understanding of LLMs as next-token prediction systems, including tokenization, and a broad sense of how pre-training data shapes model behavior.

**Programming**

Proficiency in one or more programming language(s) such as Python or R. Conceptual understanding of file input/output, API calls, data manipulation, functions, and loops. 


# Prerequisite Software & Packages
Guidance for setting up prerequisite software and packages is included in this README.md. If you have any issues with software and package set-up, please post it in [Discussions](https://github.com/hernandezb3/llm-text-classification/discussions) or email brittney.hernandez@uconn.edu.

- GitHub Account
- Hugging Face Account
- Hugging Face Access Token
- Complete Community Access Agreement for Llama 3.2
- One of **Option A** or **B**:

**Option A) Colab** *(Preferred)*
- Google Drive account

**Option B) Local**
- VS Code
- Git
- Python 3.12.3

*Note.* The session will be run in Google Colab, but Option B is included as a free option for those who might have used all of their compute on Colab or storage on Google Drive.

# Directory Structure
The directory structure is set up as follows:

```
├── llm-text-classification/
│   ├── data/
│   │   ├── dev.xlsx
│   │   ├── test.xlsx
│   │   └── train.xlsx
│   ├── data_management/
│   │   ├── human_prompt_codebook.docx
│   │   └── llm_prompt_codebook.xlsx
│   ├── results/
│   │   ├──local/
│   │   ├── non-local/
│   │   ├── prompt-engineering/
│   │   ├── fine-tuning/
│   │   └── classifications.txt
│   ├── README.md
│   ├── paths.py
│   ├── requirements.txt
│   ├── secrets-template.txt
│   ├── 00a_colab-setup.ipynb
│   ├── 00b_local-setup.ipynb
│   ├── 01_non-local.ipynb
│   ├── 02_local.ipynb
│   ├── 03_prompt-engineering.ipynb
│   └── 04_fine-tuning.ipynb
```

# Hugging Face
You will need to sign up for an account, create a read-only acess token, and complete the Community Access Agreement for Llama 3.2. 

Navigate to https://huggingface.co and click Sign Up. 

<p align="center">
<img src="readme_images/hf sign up.png" height="500">
</p>

Click on your profile icon in the top right corner and select **Access Tokens**. 
<p align="center">
<img src="readme_images/nav to tokens.png" height="500">
</p>

Click on **Create new Access Token** and on the next page click **+ Create new token**.
<p align="center">
<img src="readme_images/new token.png" height="300">
</p>

Select **Read Only**, give the token a name and click **Create Token**. A pop-up window will appear with your access token. Save this somewhere secure (e.g., a password manager). 

<p align="center">
<img src="readme_images/create token.png" height="500">
</p>

Navigate to the [Llama 3.2 1B](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) Model Card on Hugging Face and complete the Community Access Agreement.

<p align="center">
<img src="readme_images/gated access.png" height="500">
</p>

---
# Option A) Colab

Navigate to https://colab.research.google.com and click **+ New Notebook**. 

<p align="center">
<img src="readme_images/add-notebook.png" height="500">
</p>

To add a new chunk of code to your notebook click **+ Code**. Run each chunk of code by clicking the ▶️ button. 

<p align="center">
<img src="readme_images/code-chunk.png" height="200">
</p>

### Clone the GitHub repo
Copy, paste, and run the code below into a code chunk in your new Colab Notebook. It will import some packages.
```
import os
from google.colab import drive
from google.colab import userdata
```

Mount your Google Drive account to the Colab notebook. You will be prompted to sign in to Google Drive and grant the notebook access to your account.
```
drive.mount('/content/drive/')
```

Change your directory to MyDrive. 
```
os.chdir("./drive/MyDrive/")
```

Clone the GitHub reposiitory to your Google Drive.
```
!git clone https://github.com/hernandezb3/llm-text-classification.git
```

### Upload Data to Google Drive
Navigate to Google Drive. You should now see a folder called `llm-text-classification` in My Drive.

<p align="center">
<img src="readme_images/google drive.png" height="500">
</p>

Go into the `llm-text-classification/` folder and then into the the `data/` folder in 

Download the `train.xlsx`, `dev.xlsx`, and `test.xlsx` files from the `data-shared/` (linked below) and upload them to the `llm-text-classification/data/` folder.

- [data-shared/](https://uconn-my.sharepoint.com/:f:/g/personal/brittney_hernandez_uconn_edu/IgBK64FIE_1zS7aIGgZqEw0VAVnvvbCgYIYqzkKFFeRBaG4?e=e482Jo)

*Note.* The `data-shared/` folder is password-protected. You will receive an email before the training session with a 1Password Item that includes a password to the data. 

Return to your Google Colab notebook and run,
```
os.listdir("./llm-text-classification/data/")
```
Your output should include train.xlsx, dev.xlsx, and test.xlsx.

### Save Secrets
Navigate to the left panel and click on 🔑 **Secrets**. Use **+ Add new secret** to add two new secrets and `HF_TOKEN` and `OPENAI_API_KEY`. 

<img src="https://storage.googleapis.com/generativeai-downloads/images/secrets.jpg" alt="You can find the Secrets tab on the left panel." width=50%>

Add your Hugging Face token to Secrets. Name it `HF_TOKEN` and add your token to the Value column. 

Add the Open AI key to Secrets. Name it `OPENAI_API_KEY` and add the key shared in the 1Password Item. 

Toggle **Notebook access** on for both keys. 
<p align="center">
<img src="readme_images/output.png" height="200">
</p>

Check that your secrets loaded by running,
```
os.environ["HF_TOKEN"] = userdata.get('HF_TOKEN')
```
and
```
os.environ["OPENAI_API_KEY"] = userdata.get('OPENAI_API_KEY')
```

### Run a Test

Test your Hugging Face Token:
```
from huggingface_hub import login
login(token = os.getenv("HF_TOKEN"))
```

Test the Open AI Key:
```
from openai import OpenAI
client = OpenAI()
client.models.list()
```


Your output should look like this:
<p align="center">
<img src="readme_images/output.png" height="200">
</p>

---
# Option B) Local

- Download [Visual Studio Code](https://code.visualstudio.com) (VS Code)
- Download [Git](https://github.com/git-guides/install-git)
- Download [Python 3.12.3](https://www.python.org/downloads/release/python-3123/)

Clone the [llm-text-classification](https://github.com/hernandezb3/llm-text-classification) GitHub Repo to your computer.

Open your OS's Command Line Interface (Terminal for Mac or Command Prompt for Windows). Change your directory to the location where you want to save the repo.
```
cd path/to/folder/
```

Clone the repo.
```
git clone https://github.com/hernandezb3/llm-text-classification.git
```

Download the `train.xlsx`, `dev.xlsx`, and `test.xlsx` files from `data-shared` below and save them to the `data/` folder in your cloned repo. *Note* There is a .gitignore file in the data file that keeps any data from being pushed to GitHub.

- [data-shared](https://uconn-my.sharepoint.com/:f:/g/personal/brittney_hernandez_uconn_edu/IgBK64FIE_1zS7aIGgZqEw0VAVnvvbCgYIYqzkKFFeRBaG4?e=e482Jo)/

Confirm the path to the data looks like this:

```
├── llm-text-classification/
│   ├── data/
│   │   ├── dev.xlsx
│   │   ├── test.xlsx
│   │   └── train.xlsx
```
Open VS Code. Click Open, navigate to the llm-text-classification folder of the repo you just cloned, and click Open.

<p align="center">
<img src="readme_images/vs code.png" height="500">
</p>

** To continue local set up, open `00b_setup-local.ipynb` from the Explorer tab in VS Code. **

<p align="center">
<img src="readme_images/vs code explorer.png" height="500">
</p>

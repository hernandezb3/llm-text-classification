# Text Classification with Large Language Models
AIMEcon tutorial, *Text Classification with Large Language Models: Pipelines, Fine-tuning, and Measurement Validity*
By Brittney Hernandez, Claudia Ventura, Kylie Anglin

# Pre Requisites
- VS Code
- Python 3.12.2
- HuggingFace Account
- Ollama?

*optional:*
- Google Colab account
- API Key to model


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

## Tokens
### STEP B: Save credentials

To do this:
- Save credentials to a file called .env (template is in dotenv.txt)

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
- VS Code
- Python 3.12.3
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

## API Keys
These calls are made via application programming interface (API), which allow the provider to verify who is making the request (aka who to bill for it). That's where API keys come in; they're the credential that authenticates each request. 

*Secrets management* is the process of controlling IT credentials, such as API keys, passwords, and configuration files. One approach for storing secrets is as environment variables. 

### STEP B: Save credentials

To do this: 
- Create a file called `.env` in your working directory 
- Ensure that `.env` is in your `.gitignore` file
- Add your keys to `.env`:
   ```
   OPENAI_API_KEY=sk-...
   ANTHROPIC_API_KEY=sk-ant-...
   GOOGLE_API_KEY=AQ...
   ```

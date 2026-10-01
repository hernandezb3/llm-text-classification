import os
from datetime import datetime

from dotenv import load_dotenv
from pathlib import Path
from huggingface_hub import login
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from transformers import EarlyStoppingCallback

# downloads the model in 4-bit but does not quantize the lora adapters


# FILE STRUCTURE FOR HPC/COLAB
# .env in cd
# data to data/
# prompt_codebook to data_management/
# classifications.xlsx to results/
# make sure results/local exists
start_time = datetime.now()
print(f"finetuning started: {start_time}\n")

load_dotenv()

USER = "hpc" 

if USER == "local":
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
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL = "meta-llama/Llama-3.2-1B-Instruct"
PROMPT_ID = "prompt3_c5t1d4g6"
RANDOM_STATE = 42
TASK_TYPE = "CAUSAL_LM"
LEARN_RATE = 2e-4
TRAIN_SAMPLE_NAME = "train_balanced"
DROP_LONG_CASES = False
# based on variables ^^ add a description/id to the ft model
DESCRIPTION = "bestPrompt_baseConfig"

print(f"\nUsing device: {DEVICE}")
if DEVICE == "cuda":
    if torch.cuda.is_bf16_supported():
        PRECISION = torch.bfloat16
    else:
        PRECISION = torch.float16
    print(torch.cuda.get_device_name(0))
else:
    raise RuntimeError("No GPU found: 4-bit QLoRA training requires CUDA.")

# float32 = full precision for CPU
# can use bfloat16 if cuda is available (float for cpu, bfloat for gpu)

# ---- get data ----
path_to_train = DATA_DIR / "train.xlsx"
train = pd.read_excel(path_to_train)
train["word_counts"] = train["text"].str.split(" ").str.len()

# remove long utterances - crashing 3B FT
# train["word_counts"].describe(percentiles=[.5, .9, .99, .999])
if DROP_LONG_CASES:
    train = train[train["word_counts"] <= 700]

max_utterance_len = train["word_counts"].max()

path_to_dev = DATA_DIR / "dev.xlsx"
dev = pd.read_excel(path_to_dev)
#df = df.sample(n = 5, ignore_index = True)
# call out in the room, what performance did you estimate
# performance metrics are estimates > seguey to uncertainty

not_prompt_train = train[train["code_human"] == 0]
prompt_train = train[train["code_human"] == 1]
minority_rows = min(len(not_prompt_train), len(prompt_train))

train_balanced = pd.concat([
    not_prompt_train.sample(n = minority_rows, random_state = RANDOM_STATE),
    prompt_train.sample(n = minority_rows, random_state = RANDOM_STATE), 
    ]).sample(frac = 1, random_state = RANDOM_STATE).reset_index(drop = True)

print("\nSAMPLE SIZE\n")
print(f"n train: {len(train):,}\n{train['code_human'].value_counts()}\n")
print(f"n train balanced: {len(train_balanced):,}\n{train_balanced['code_human'].value_counts()}\n")
print(f"n dev: {len(dev):,}\n{dev['code_human'].value_counts()}\n")

# ---- get prompt codebook ----
# path_to_prompts = WORKING_DIR / "data_management" / "llm_codebook.xlsx"
# prompts = pd.read_excel(path_to_prompts)
# prompt_dictionary = prompts.set_index("id")["prompt"].to_dict()

# def prompt_case(case, p):
#     parts = [
#         p["task_1"],
#         p["definition_1"],
#         p["format_case"],
#         ]
    
#     parts.append(f"\n\"\"\"{case}\"\"\"\n")
#     parts.append(p["format_local"])

#     return "\n\n".join(part for part in parts if part)


# empirical prompt:
path_to_prompts = WORKING_DIR / "data_management" / "empirical_prompts_50.csv"
prompts = pd.read_csv(path_to_prompts).fillna("")
prompt_dictionary = prompts.set_index("prompt_id").to_dict(orient = "index")

def prompt_case(case, p):
    parts = [
        p["context"],
        p["task"],
        p["definition"],
    ]

    # only include the guidance header if any guidance was drawn
    if len(p["guidance"]) > 0:
        parts.append(p["format_guidance"])
        parts.append(p["guidance"].strip())

    parts.append(p["format_case"])
    parts.append(f"\n\"\"\"{case}\"\"\"\n")                 # <- the case goes here
    parts.append(p["format_local"])    # e.g. output format instructions

    return "\n\n".join(part for part in parts if part)

# choose which prompt to use
prompt_id = PROMPT_ID

prompt_template = prompt_case("CASE", prompt_dictionary[prompt_id])
print(f"\n\nPROMPT TEMPLATE: {prompt_id}\n{prompt_template}\n\n")

prompt_len = len(prompt_template.split(" "))


# ---- set up model ----
login(token = os.getenv("HF_TOKEN"))

# finetuning
# meta-llama/Llama-3.2-1B-Instruct
# meta-llama/Llama-3.2-3B-Instruct
# ibm-granite/granite-4.2-3b 
# Qwen/Qwen2.5-7B-Instruct
# ibm-granite/granite-4.2-8b 


# ON COLAB
# meta-llama/Llama-3.2-1B-Instruct (baseline) x

bnb_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_quant_type = "nf4",
    bnb_4bit_compute_dtype = PRECISION,
    bnb_4bit_use_double_quant = True,
    )


# make sure permissions are on 
#client = transformers.pipeline(TASK, model = MODEL, 
#                               model_kwargs = {"dtype": PRECISION}, 
#                               token = os.getenv("HF_TOKEN")) # initate pipeline

model = AutoModelForCausalLM.from_pretrained(MODEL, 
                                             quantization_config = bnb_config,
                                             dtype = PRECISION,
                                             token = os.getenv("HF_TOKEN"),
                                             device_map = "auto") # initate pipeline

# caches attention key/values. faster generation but conflic w gradient checkpoint
model.config.use_cache = False

hf_tokenizer = AutoTokenizer.from_pretrained(MODEL, token = os.getenv("HF_TOKEN"))

# default padding = False, side is determined from class, typically right
hf_tokenizer.padding_side = "right" 
model.config.pad_token_id = hf_tokenizer.eos_token_id

# see how much space the model occupies in memory
print(model.get_memory_footprint() / 1e9, "GB")

if getattr(model.config, "quantization_config", None) is not None:
    QUANTIZATION = model.config.quantization_config
else:
    QUANTIZATION = None

print("\n\nFINETUNING INFO")
print(f"Model: {MODEL} \nPrecision: {PRECISION} \nQuantization: {QUANTIZATION}")

# see the layers in the model
# print(model)

# prep data for finetuning
# tokenize prompt

# claudia look here -- this function replaces the commented out function below
# the completion is just the label and then i add completion_only_loss to the sft config
def make_prompt_completion(row, tokenizer):
    label_str = "Yes" if row["code_human"] == 1 else "No"

    return {
        "prompt": [
            {"role": "user", "content": prompt_case(str(row["text"]), prompt_dictionary[prompt_id])},
            ],

        "completion": [
            {"role": "assistant", "content": label_str},
            ],
        }

#def make_prompt_completion(row, tokenizer):
#    label_str = "Yes" if row["code_human"] == 1 else "No"
#    messages = [
#        {"role": "system",    "content": "You are a helpful text classifier."},
#        {"role": "user",      "content": build_user_prompt(str(row["text"]))},
#        {"role": "assistant", "content": label_str},
#    ]
#    return {"text": tokenizer.apply_chat_template(
#        messages, tokenize=False, add_generation_prompt=False
#    )}


samples = {"train": train, "train_balanced": train_balanced}

TRAIN_SAMPLE = samples[TRAIN_SAMPLE_NAME]

train_hf = Dataset.from_list([
     make_prompt_completion(row, hf_tokenizer) for _, row in TRAIN_SAMPLE.iterrows()
     ])

dev_hf = Dataset.from_list([
    make_prompt_completion(row, hf_tokenizer) for _, row in dev.iterrows()
    ])

lora_config = LoraConfig(
    r = 16, # rank of the adapter TRY: 8 
    lora_alpha = 32, # multiplier, usually 2*r
    lora_dropout = 0.05, # TRY: more regularization .10
    bias = "none",
    task_type = TASK_TYPE,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
)

model = prepare_model_for_kbit_training(model, use_gradient_checkpointing = True)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


path_to_output = RESULTS_DIR / "finetune"
sft_config = SFTConfig(
    output_dir = path_to_output,
    completion_only_loss = True, # claudia look here: this restricts the loss to the completion tokens aka the y/n label
    num_train_epochs = 3,
    per_device_train_batch_size = 16,
    per_device_eval_batch_size = 32,
    gradient_accumulation_steps = 1,
    warmup_steps = 10,
    learning_rate = LEARN_RATE, # TRY: slow down learning rate 1e-4 or 5e-5
    max_grad_norm = 0.3,
    fp16 = PRECISION == torch.float16,
    bf16 = PRECISION == torch.bfloat16,
    logging_steps = 10,
    eval_strategy = "steps", 
    eval_steps = 20,
    save_strategy = "steps", 
    save_steps = 20,
    save_total_limit = 3,
    load_best_model_at_end = True,
    metric_for_best_model = "eval_loss", # TRY? f1 or kappa
    # greater_is_better = True, # need this if using f1 or kappa
    report_to = "none",
    max_length = int((max_utterance_len + prompt_len) * 1.5) + 150, # claudia look here
    optim = "paged_adamw_8bit", # "adamw_torch",
)

trainer = SFTTrainer(
    model = model,
    args = sft_config,
    train_dataset = train_hf,
    eval_dataset = dev_hf,
    processing_class = hf_tokenizer,
    callbacks = [EarlyStoppingCallback(early_stopping_patience = 6)]
)

# print config before finetuning
run_info = f"""Finetuning run info\n\n
- Description: {DESCRIPTION}
- Started: {start_time}
- Prompt: {PROMPT_ID}
- Sample: {TRAIN_SAMPLE_NAME} 
- Dropped long cases: {DROP_LONG_CASES}
- n = {len(TRAIN_SAMPLE):,}
- Model: {MODEL}
- Precision: {PRECISION}
- Task Type: {TASK_TYPE}
- Learning Rate: {LEARN_RATE}
- LoRA: r = {lora_config.r}, alpha = {lora_config.lora_alpha}, dropout = {lora_config.lora_dropout}
- Max length: {sft_config.max_length}"""

print("\n\n" + run_info)

trainer.train()

results = f"""\n\nTraining Results:\n\n
- Best checkpoint: {trainer.state.best_metric}
- Best loss: {trainer.state.best_metric}
"""

path_to_finetuned_model = RESULTS_DIR / f'{MODEL}_FT_{DESCRIPTION}'
trainer.save_model(path_to_finetuned_model)

with open(path_to_finetuned_model / "README.md", "w", encoding = "utf-8") as f:
    f.write("\n" + run_info + "\n" + results)

# trainer.push_to_hub()
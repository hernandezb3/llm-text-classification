# test local setup
import os
from dotenv import load_dotenv

load_dotenv()

from huggingface_hub import login, HfApi
login(token = os.getenv("HF_TOKEN"))
client = HfApi()
client.model_info("meta-llama/Llama-3.2-1B")

from openai import OpenAI
client = OpenAI()
client.models.list()

print("\n✅ Setup Complete\n")
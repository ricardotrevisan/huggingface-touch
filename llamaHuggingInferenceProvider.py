# Inference Provider: Novita
# Model: meta-llama/Llama-3.2-3B-Instruct

import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv  # Add this import

load_dotenv()  # Load environment variables from .env file

client = InferenceClient(
    provider="novita",
    api_key=os.environ["HF_TOKEN"],
)

completion = client.chat.completions.create(
    model="meta-llama/Llama-3.2-3B-Instruct",
    messages=[
        {
            "role": "user",
            "content": "What is the capital of France?"
        }
    ],
)

print(completion.choices[0].message)
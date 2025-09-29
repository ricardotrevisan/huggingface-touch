from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 1️⃣ Configurações
model_id = "meta-llama/Llama-3.2-3B-Instruct"
token = ""  # seu token Hugging Face

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Usando dispositivo:", device)

# 2️⃣ Carrega tokenizer e modelo (direto na GPU)
tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    token=token,
    device_map="auto",
    torch_dtype=torch.float16
)

# model.to("cuda")

# 3️⃣ Prompt de teste
prompt = "Explique rapidamente o que é PyTorch."
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# 4️⃣ Gera a resposta
with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7
    )

# 5️⃣ Decodifica
response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("Resposta do modelo:\n", response)

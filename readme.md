### Hugging Face Model Integrations

## Models
## Datasets
## Spaces (hosting apps)

## Requisites
1. PyTorch installation (GPU )
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

```

## Login
1. token creation
1. huggingface-cli login
1. paste token


---
## WSL Setup (RTX3080=Desktop) : LLAMA 
python3 -m venv .venv
source .venv/bin/activate

### Prospect LLM Model https://huggingface.co/bartowski/Qwen_Qwen3-30B-A3B-GGUF/blob/main/Qwen_Qwen3-30B-A3B-Q4_K_S.gguf

Download the .gguf (quantization)
```
huggingface-cli download bartowski/Qwen_Qwen3-30B-A3B-GGUF Qwen_Qwen3-30B-A3B-Q4_K_S.gguf --local-dir ./model --local-dir-use-symlinks False
```

Environment
```
sudo apt install nvidia-cuda-toolkit
```

Path:
```
vi ~/.bashrc \
    export PATH=/usr/local/cuda/bin:$PATH \
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```
Requirements:
```
sudo apt install libcurl4-openssl-dev
```

Make & Build:
```
cmake -B build -DLLAMA_CUDA=ON -DLLAMA_CURL=OFF
cmake --build build --config Release -j
```

Run:
CLI
```
./build/bin/llama-run --ngl 32 ./model/Qwen_Qwen3-30B-A3B-Q4_K_S.gguf "Hello, who are you?"
```

Server (http://127.0.0.1:8080/):
* Ollama port conflict
```
export GGML_USE_CUBLAS=1
export GGML_USE_CLBLAST=0
export GGML_USE_ACCELERATE=0
./build/bin/llama-server --model ./model/Qwen_Qwen3-30B-A3B-Q4_K_S.gguf --port 11435 --ngl 32
```
curl GET:
```
curl -X POST http://127.0.0.1:11435/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "./model/Qwen_Qwen3-30B-A3B-Q4_K_S.gguf",
        "prompt": "Hello, who are you?",
        "max_tokens": 256
      }'
```
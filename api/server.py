from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Nome do diretório onde salvamos o modelo treinado
MODEL_PATH = "../model/falcon-7b-lora-finetuned"

# Carregar o modelo e o tokenizador
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,  # Usa FP16 para reduzir memória
    device_map="auto"  # Distribui automaticamente entre CPU e GPU
)

# Criar a API FastAPI
app = FastAPI(title="API do Falcon 7B", version="1.0")

# Rota para verificar se a API está funcionando
@app.get("/")
def home():
    return {"message": "API do Falcon 7B está rodando!"}

# Rota para gerar texto
@app.post("/generate/")
def generate_text(prompt: str, max_length: int = 100, temperature: float = 0.7):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True
        )
    
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return {"prompt": prompt, "generated_text": generated_text}

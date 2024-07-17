from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Cargar el modelo preentrenado y el tokenizador
model_name = "gpt2"  # Puedes usar otros modelos como "gpt2-medium", "gpt2-large", etc.
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

def generate_text(prompt):
    # Tokenizar el prompt
    inputs = tokenizer.encode(prompt, return_tensors="pt")

    # Crear una máscara de atención
    attention_mask = torch.ones(inputs.shape, dtype=torch.long)

    # Generar texto
    outputs = model.generate(
        inputs,
        max_length=50,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,  # Configurar pad_token_id al eos_token_id
        attention_mask=attention_mask,        # Configurar explicitamente la máscara de atención
    )

    # Decodificar el texto generado
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return text

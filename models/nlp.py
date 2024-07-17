from transformers import pipeline, set_seed
import random

def generate_text(prompt):
    generator = pipeline('text-generation', model='gpt2')
    set_seed(random.randint(1, 10000))  # Para variedad en las generaciones
    result = generator(prompt, max_length=50, num_return_sequences=3)
    
    # Seleccionar la mejor respuesta (puedes implementar una lógica más sofisticada aquí)
    best_response = max(result, key=lambda x: len(x['generated_text']))
    
    return best_response['generated_text'].strip()
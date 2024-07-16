import os
from openai import OpenAI, OpenAIError
from dotenv import load_dotenv

# Cargar variables de entorno desde el archivo .env
load_dotenv()

# Configuración de la API de OpenAI
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

def generate_text(prompt):
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="gpt-3.5-turbo",
        )
        return chat_completion.choices[0].message['content']
    except OpenAIError as e:
        print(f"OpenAIError: {e}")
        return "Error: No se pudo generar el texto debido a un problema con la cuota de la API."
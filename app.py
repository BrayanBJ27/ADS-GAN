import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import tensorflow as tf
import os
from models.generator import build_generator
from utils.data_loader import save_images
from models.nlp import generate_text  # Importar la función desde nlp.py

# Configuraciones del generador
SEED_SIZE = 100
IMAGE_SIZE = 470
CHANNELS = 3

# Crear el modelo generador
generator = build_generator(SEED_SIZE, CHANNELS)
generator.load_weights('generator_weights.weights.h5')  # Ruta a los pesos entrenados del generador

# Función para generar una imagen del anuncio
def generate_ad(prompt, color, punchline, punchline_color, button_text, button_color, base_image, logo_image):
    # Generar la imagen base con la GAN
    seed = np.random.normal(0, 1, (1, SEED_SIZE))
    generated_image = generator.predict(seed)
    generated_image = 0.5 * generated_image + 0.5
    generated_image = (generated_image[0] * 255).astype(np.uint8)
    generated_image = Image.fromarray(generated_image)

    # Redimensionar la imagen base y el logo
    base_image = base_image.resize((IMAGE_SIZE, IMAGE_SIZE))
    logo_image = logo_image.resize((100, 100))  # Tamaño fijo para el logo

    # Combinar las imágenes
    ad_image = Image.alpha_composite(base_image.convert("RGBA"), generated_image.convert("RGBA"))
    ad_image.paste(logo_image, (ad_image.width - logo_image.width - 10, 10), logo_image)  # Posicionar el logo

    # Dibujar el punchline y el botón
    draw = ImageDraw.Draw(ad_image)
    font = ImageFont.load_default()

    # Dibujar punchline
    draw.text((10, 10), punchline, fill=punchline_color, font=font)

    # Dibujar botón
    button_width, button_height = draw.textsize(button_text, font=font)
    draw.rectangle([10, ad_image.height - 30, 10 + button_width, ad_image.height - 10], fill=button_color)
    draw.text((10, ad_image.height - 30), button_text, fill=(255, 255, 255), font=font)

    return ad_image

# Definir la interfaz de usuario
st.title("Generador de Anuncios Publicitarios")

prompt = st.text_input("Prompt:", "Ingrese un prompt para el anuncio")
color = st.color_picker("Color para la imagen generada:", "#ffffff")
punchline = st.text_input("Punchline:")
punchline_color = st.color_picker("Color del punchline:", "#000000")
button_text = st.text_input("Texto del botón:")
button_color = st.color_picker("Color del botón:", "#00ff00")

base_image = st.file_uploader("Suba una imagen base:", type=["jpg", "jpeg", "png"])
logo_image = st.file_uploader("Suba un logo con fondo blanco:", type=["jpg", "jpeg", "png"])

if st.button("Generar Anuncio"):
    if base_image is not None and logo_image is not None:
        base_image = Image.open(base_image)
        logo_image = Image.open(logo_image)
        
        # Obtener texto generado por OpenAI
        generated_punchline = generate_text(prompt)
        
        ad_image = generate_ad(prompt, color, generated_punchline, punchline_color, button_text, button_color, base_image, logo_image)
        
        # Mostrar la imagen generada debajo del botón
        st.image(ad_image, caption="Anuncio Generado", width=IMAGE_SIZE)
    else:
        st.error("Por favor, suba una imagen base y un logo.")
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
generator.load_weights('ad-gen/generator_weights.weights.h5')  # Ruta a los pesos entrenados del generador

# Función para generar una imagen del anuncio
def generate_ad(prompt, color, punchline, punchline_color, button_text, button_color, base_image, logo_image):
    # Generar la imagen base con la GAN
    seed = np.random.normal(0, 1, (1, SEED_SIZE))
    generated_image = generator.predict(seed)
    generated_image = 0.5 * generated_image + 0.5
    generated_image = (generated_image[0] * 255).astype(np.uint8)
    generated_image = Image.fromarray(generated_image)

    # Redimensionar la imagen base y el logo
    base_image = base_image.resize((IMAGE_SIZE, IMAGE_SIZE)).convert("RGBA")
    generated_image = generated_image.resize((IMAGE_SIZE, IMAGE_SIZE)).convert("RGBA")
    logo_image = logo_image.resize((100, 100)).convert("RGBA")  # Tamaño fijo para el logo

    # Combinar las imágenes
    ad_image = Image.alpha_composite(base_image, generated_image)
    ad_image.paste(logo_image, (ad_image.width - logo_image.width - 10, 10), logo_image)  # Posicionar el logo

    # Dibujar el punchline y el botón
    draw = ImageDraw.Draw(ad_image)
    font = ImageFont.load_default()

    # Dibujar punchline
    draw.text((10, 10), punchline, fill=punchline_color, font=font)

    # Obtener dimensiones del texto del botón
    button_bbox = draw.textbbox((10, ad_image.height - 30), button_text, font=font)
    button_width = button_bbox[2] - button_bbox[0]
    button_height = button_bbox[3] - button_bbox[1]

    # Dibujar botón
    draw.rectangle([10, ad_image.height - 30, 10 + button_width, ad_image.height - 10], fill=button_color)
    draw.text((10, ad_image.height - 30), button_text, fill=(255, 255, 255), font=font)

    return ad_image

# Definir la interfaz de usuario
st.set_page_config(layout="wide")  # Configurar la página a ancho completo

# Sidebar
st.sidebar.title("PUBLICIDAD - GAN")
st.sidebar.write("¡La publicidad generativa es el futuro de la publicidad!")
st.sidebar.subheader("Ejemplo de Entradas:")
st.sidebar.markdown("**Prompt:**\n- Una taza de café de cartón en la mesa")
st.sidebar.markdown("**Frase:**\n- La creatividad es ver lo que otros ven y pensar lo que nadie más ha pensado.")
st.sidebar.markdown("**Botón:**\n- ¡La filosofía es divertida!")
st.sidebar.markdown("**Color:**\n- Rojo")

# Parámetros
# Parámetros
st.markdown("<h1 style='text-align: center;'>ANUNCIOS PUBLICITARIOS - GAN</h1>", unsafe_allow_html=True)
st.markdown("""---""")
st.subheader("Parámetros")

col1, col2, col3 = st.columns([2, 1, 2])

with col1:
    prompt = st.text_area("Prompt:", "Ingrese un prompt para el anuncio")
    punchline = st.text_area("Frase:")
    button_text = st.text_area("Texto del botón:")

with col2:
    color = st.color_picker("Color para la imagen generada:", "#ffffff")
    punchline_color = st.color_picker("Color de la frase:", "#000000")
    button_color = st.color_picker("Color del botón:", "#00ff00")

with col3:
    st.write("Vista previa de la imagen base:")
    base_image = st.file_uploader("Suba una imagen base:", type=["jpg", "jpeg", "png"])
    if base_image:
        st.image(Image.open(base_image), width=150)
    st.write("Vista previa del logo:")
    logo_image = st.file_uploader("Suba un logo con fondo blanco:", type=["jpg", "jpeg", "png"])
    if logo_image:
        st.image(Image.open(logo_image), width=150)

if st.button("¡Generar Anuncio!"):
    if base_image is not None and logo_image is not None:
        base_image = Image.open(base_image)
        logo_image = Image.open(logo_image)
        
        # Obtener texto generado localmente
        generated_punchline = generate_text(prompt)
        
        ad_image = generate_ad(prompt, color, generated_punchline, punchline_color, button_text, button_color, base_image, logo_image)
        
        # Mostrar la imagen generada debajo del botón
        st.image(ad_image, caption="Anuncio Generado", width=IMAGE_SIZE)
    else:
        st.error("Por favor, suba una imagen base y un logo.")

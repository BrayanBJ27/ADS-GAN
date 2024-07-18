# Generador de Anuncios Publicitarios

Este proyecto utiliza GANs (Generative Adversarial Networks) para generar anuncios publicitarios a partir de un prompt dado por el usuario y varios parámetros personalizados.

# Integrantes
- Davila Brayan
- Garzon Edgar
- Mendez Pablo
- Pallo Christopher

## Instrucciones de Uso

1. Instala las dependencias:
   
```sh
pip install -r requirements.txt
```


3. Entrenamiento del Modelo

```sh
python train_gan.py
```

4. Ejecuta la aplicación:
   
```sh
streamlit run app.py
```

## Estructura del Proyecto
- `ad-gen`: Carpeta donde se guarda en archivo binario de entrenaminto.
- `app.py`: Interfaz de usuario con Streamlit.
- `models/`: Contiene los modelos de GAN.
- `utils/`: Contiene utilidades para cargar datos y preprocesamiento.
- `requiriments`: Contiene las librerías y dependencias requeridas.

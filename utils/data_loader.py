import os
from PIL import Image
import numpy as np

def load_images(image_folder, image_size):
    images = []
    for filename in os.listdir(image_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img = Image.open(os.path.join(image_folder, filename)).resize((image_size, image_size))
            img = np.array(img) / 255.0
            images.append(img)
    return np.array(images)

def save_images(cnt, noise, generator, output_path, generate_square, image_channels):
    image_array = np.full((generate_square, generate_square, image_channels), 255, dtype=np.uint8)

    generated_images = generator.predict(noise)
    generated_images = 0.5 * generated_images + 0.5

    image_array[0:generate_square, 0:generate_square] = generated_images[0] * 255

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    filename = os.path.join(output_path, f"train-{cnt}.png")
    im = Image.fromarray(image_array)
    im.save(filename)

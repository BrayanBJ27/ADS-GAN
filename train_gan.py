import tensorflow as tf
from tensorflow.keras.layers import Input, Reshape, Dropout, Dense, Flatten, BatchNormalization, Activation, ZeroPadding2D, LeakyReLU, UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np
from PIL import Image
from tqdm import tqdm
import os
import time
import matplotlib.pyplot as plt

# Nicely formatted time string
def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)

# Configuraciones
GENERATE_SQUARE = 470  # Dimensión deseada
IMAGE_CHANNELS = 3
SEED_SIZE = 100
EPOCHS = 50
BATCH_SIZE = 32
BUFFER_SIZE = 60000
DATA_PATH = 'images'  # Asegúrate de tener una carpeta llamada 'images' con imágenes de anuncios publicitarios

# Preparación y procesamiento de los datos
training_binary_path = os.path.join(DATA_PATH, f'training_data_{GENERATE_SQUARE}_{GENERATE_SQUARE}.npy')
print(f"Looking for file: {training_binary_path}")

if not os.path.isfile(training_binary_path):
    start = time.time()
    print("Loading training images...")

    training_data = []
    ad_path = os.path.join(DATA_PATH)
    for filename in tqdm(os.listdir(ad_path)):
        path = os.path.join(ad_path, filename)
        image = Image.open(path)
        if image.size != (GENERATE_SQUARE, GENERATE_SQUARE):
            image = image.resize((GENERATE_SQUARE, GENERATE_SQUARE), Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        if image_array.shape == (GENERATE_SQUARE, GENERATE_SQUARE, IMAGE_CHANNELS):
            training_data.append(image_array)
        else:
            print(f"Skipping image {filename} due to incorrect shape: {image_array.shape}")

    training_data = np.array(training_data)
    training_data = training_data.astype(np.float32)
    training_data = training_data / 127.5 - 1.

    print("Saving training image binary...")
    np.save(training_binary_path, training_data)
    elapsed = time.time() - start
    print(f'Image preprocess time: {hms_string(elapsed)}')
else:
    print("Loading previous training binary...")
    training_data = np.load(training_binary_path)

# Verificar que training_data no esté vacío
print(f"Training data shape: {training_data.shape}")

# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(training_data).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# Verificar que train_dataset no esté vacío
for batch in train_dataset:
    print(f"Batch shape: {batch.shape}")
    break

# Creación del generador
def build_generator(seed_size, channels):
    model = Sequential()
    model.add(Dense(15*15*256, activation="relu", input_dim=seed_size))
    model.add(Reshape((15, 15, 256)))

    model.add(UpSampling2D())
    model.add(Conv2D(256, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    model.add(UpSampling2D())
    model.add(Conv2D(256, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    if 470 > 1:
        model.add(UpSampling2D(size=(15, 15)))
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

    model.add(Conv2D(channels, kernel_size=3, padding="same"))
    model.add(Activation("tanh"))

    return model

# Creación del discriminador
def build_discriminator(image_shape):
    model = Sequential()

    model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=image_shape, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Conv2D(512, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    return model

# Guardar imágenes generadas
def save_images(cnt, noise):
    image_array = np.full((GENERATE_SQUARE, GENERATE_SQUARE, IMAGE_CHANNELS), 255, dtype=np.uint8)

    generated_images = generator.predict(noise)
    generated_images = 0.5 * generated_images + 0.5

    image_array[0:GENERATE_SQUARE, 0:GENERATE_SQUARE] = generated_images[0] * 255

    output_path = os.path.join(DATA_PATH, 'output')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    filename = os.path.join(output_path, f"train-{cnt}.png")
    im = Image.fromarray(image_array)
    im.save(filename)

# Creación del generador y discriminador
generator = build_generator(SEED_SIZE, IMAGE_CHANNELS)
discriminator = build_discriminator((GENERATE_SQUARE, GENERATE_SQUARE, IMAGE_CHANNELS))

# Definir la función de pérdida y los optimizadores
cross_entropy = tf.keras.losses.BinaryCrossentropy()

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1.5e-4, 0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(1.5e-4, 0.5)

# Función de entrenamiento
@tf.function
def train_step(images):
    seed = tf.random.normal([BATCH_SIZE, SEED_SIZE])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(seed, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss

# Función principal de entrenamiento
def train(dataset, epochs):
    fixed_seed = np.random.normal(0, 1, (1, SEED_SIZE))
    start = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()
        gen_loss_list = []
        disc_loss_list = []

        for image_batch in dataset:
            t = train_step(image_batch)
            gen_loss_list.append(t[0])
            disc_loss_list.append(t[1])

        if gen_loss_list and disc_loss_list:
            g_loss = sum(gen_loss_list) / len(gen_loss_list)
            d_loss = sum(disc_loss_list) / len(disc_loss_list)

            epoch_elapsed = time.time() - epoch_start
            print(f'Epoch {epoch + 1}, gen loss={g_loss}, disc loss={d_loss}, {hms_string(epoch_elapsed)}')
            save_images(epoch, fixed_seed)
        else:
            print(f'Epoch {epoch + 1} skipped due to empty loss list.')

    elapsed = time.time() - start
    print(f'Training time: {hms_string(elapsed)}')

# Entrenar el modelo
train(train_dataset, EPOCHS)

# Al final del script de entrenamiento, guarda los pesos en el directorio 'ad-gen'
weights_path = os.path.join('generator_weights.weights.h5')
generator.save_weights(weights_path)

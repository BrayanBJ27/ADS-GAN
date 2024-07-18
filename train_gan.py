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
IMAGE_CHANNELS = 4  # Cambiado a 4 para manejar RGBA
SEED_SIZE = 100
EPOCHS = 100
BATCH_SIZE = 32
SAVE_INTERVAL = 50 #Guarda Puntos de Control
BUFFER_SIZE = 60000
DATA_PATH = 'images'

# Verdades básicas adversarias
valid = np.ones((BATCH_SIZE, 1))
fake = np.zeros((BATCH_SIZE, 1))

# Preparación y procesamiento de los datos
training_binary_path = os.path.join(DATA_PATH, f'training_data_{GENERATE_SQUARE}_{GENERATE_SQUARE}.npy')
print(f"Looking for file: {training_binary_path}")

if not os.path.isfile(training_binary_path):
    start = time.time()
    print("Loading training images...")

    training_data = []
    ad_path = os.path.join(DATA_PATH)
    total_files = len([name for name in os.listdir(ad_path) if os.path.isfile(os.path.join(ad_path, name))])
    print(f"Total files found: {total_files}")

    for filename in tqdm(os.listdir(ad_path)):
        path = os.path.join(ad_path, filename)
        try:
            image = Image.open(path)
            if image.size != (GENERATE_SQUARE, GENERATE_SQUARE):
                image = image.resize((GENERATE_SQUARE, GENERATE_SQUARE), Image.Resampling.LANCZOS)
            # Convertir a RGB si es necesario
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image_array = np.asarray(image)
            if image_array.shape == (GENERATE_SQUARE, GENERATE_SQUARE, 3):
                training_data.append(image_array)
            else:
                print(f"Skipping image {filename} due to incorrect shape: {image_array.shape}")
        except Exception as e:
            print(f"Error loading image {filename}: {str(e)}")

    print(f"Successfully loaded {len(training_data)} images")

    if len(training_data) > 0:
        training_data = np.array(training_data)
        training_data = training_data.astype(np.float32)
        training_data = training_data / 127.5 - 1.

        print("Saving training image binary...")
        np.save(training_binary_path, training_data)
        elapsed = time.time() - start
        print(f'Image preprocess time: {hms_string(elapsed)}')
    else:
        print("No valid images found. Please check your data directory.")
        exit(1)
else:
    print("Loading previous training binary...")
    training_data = np.load(training_binary_path)

print(f"Final training data shape: {training_data.shape}")

# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(training_data).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# Verificar que train_dataset no esté vacío
for batch in train_dataset:
    print(f"Batch shape: {batch.shape}")
    break

def build_generator(seed_size, channels):
    model = Sequential()
    model.add(Input(shape=(seed_size,)))
    model.add(Dense(30*30*256, activation="relu"))
    model.add(Reshape((30, 30, 256)))

    model.add(UpSampling2D())
    model.add(Conv2D(256, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    model.add(UpSampling2D())
    model.add(Conv2D(64, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    model.add(UpSampling2D(size=(2,2)))
    model.add(Conv2D(channels, kernel_size=3, padding="same"))
    model.add(Activation("tanh"))

    # Crop to 470x470
    model.add(tf.keras.layers.Cropping2D(cropping=((5, 5), (5, 5))))

    return model

def build_discriminator(image_shape):
    model = Sequential()

    model.add(Conv2D(64, kernel_size=3, strides=2, input_shape=image_shape, padding="same"))
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

def save_images(cnt, noise):
    generated_images = generator.predict(noise)
    generated_images = 0.5 * generated_images + 0.5

    for i in range(generated_images.shape[0]):
        image = generated_images[i] * 255
        image = image.astype(np.uint8)
        im = Image.fromarray(image)
        
        output_path = os.path.join(DATA_PATH, 'output')
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        filename = os.path.join(output_path, f"train-{cnt}-{i}.png")
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

# Configuración de checkpoints
checkpoint_dir = 'checkpoints'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# Buscar el último checkpoint
checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('gan_epoch_') and f.endswith('.weights.h5')]
latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[2].split('.')[0])) if checkpoints else None

# Inicializar la época inicial
initial_epoch = 0

if latest_checkpoint:
    latest_checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
    print("Checkpoint encontrado:", latest_checkpoint_path)
    # Cargar los pesos del generador
    generator.load_weights(latest_checkpoint_path)
    
    # Extraer el número de época del nombre del archivo
    initial_epoch = int(latest_checkpoint.split('_')[2].split('.')[0])
    print(f"Continuando el entrenamiento desde la época {initial_epoch}")
else:
    print("No se ha encontrado ningún checkpoint, empezando desde cero.")

# Función de entrenamiento
@tf.function
def train_step(images):
    seed = tf.random.normal([BATCH_SIZE, SEED_SIZE])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(seed, training=True)
        
        # Asegúrate de que las imágenes generadas tengan el tamaño correcto
        generated_images = tf.image.resize(generated_images, [GENERATE_SQUARE, GENERATE_SQUARE])

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

    for epoch in range(initial_epoch, epochs):
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

            # Guardar checkpoint
            if (epoch + 1) % SAVE_INTERVAL == 0:
                generator.save_weights(f"{checkpoint_dir}/gan_epoch_{epoch + 1}.weights.h5")
        else:
            print(f'Epoch {epoch + 1} skipped due to empty loss list.')

    elapsed = time.time() - start
    print(f'Training time: {hms_string(elapsed)}')

# Entrenar el modelo
train(train_dataset, EPOCHS + initial_epoch)

# Al final del script de entrenamiento, guarda los pesos en el directorio 'ad-gen'
weights_path = os.path.join('ad-gen', 'generator_weights.weights.h5')
generator.save_weights(weights_path)
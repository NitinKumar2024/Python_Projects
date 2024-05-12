import numpy as np
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define constants
IMAGE_SIZE = (64, 64)
BATCH_SIZE = 4
NUM_EPOCHS = 100
LATENT_DIM = 100
NUM_NITIN_IMAGES = 11
NUM_MOUNTAIN_IMAGES = 11

# Load and preprocess dataset
data_generator = ImageDataGenerator(rescale=1. / 255)
train_data = data_generator.flow_from_directory(
    'F:\\Python_Problems\\Dataset\\Image_Dataset',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode=None,
    classes=['Nitin', 'mountain']
)

# Define generator model (same as before)

# Define discriminator model (same as before)

# Compile discriminator (same as before)

# Compile combined model (same as before)

# Define generator model
generator = models.Sequential([
    layers.Dense(8 * 8 * 256, input_shape=(LATENT_DIM,)),
    layers.Reshape((8, 8, 256)),
    layers.Conv2DTranspose(128, kernel_size=3, strides=2, padding='same'),
    layers.BatchNormalization(),
    layers.LeakyReLU(alpha=0.2),
    layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding='same'),
    layers.BatchNormalization(),
    layers.LeakyReLU(alpha=0.2),
    layers.Conv2DTranspose(3, kernel_size=3, strides=2, padding='same', activation='sigmoid')
])

# Define discriminator model
discriminator = models.Sequential([
    layers.Conv2D(64, kernel_size=3, strides=2, padding='same', input_shape=IMAGE_SIZE + (3,)),
    layers.LeakyReLU(alpha=0.2),
    layers.Dropout(0.4),
    layers.Conv2D(128, kernel_size=3, strides=2, padding='same'),
    layers.BatchNormalization(),
    layers.LeakyReLU(alpha=0.2),
    layers.Dropout(0.4),
    layers.Conv2D(256, kernel_size=3, strides=2, padding='same'),
    layers.BatchNormalization(),
    layers.LeakyReLU(alpha=0.2),
    layers.Dropout(0.4),
    layers.Flatten(),
    layers.Dense(1, activation='sigmoid')
])

# Compile discriminator
discriminator.compile(optimizer=optimizers.Adam(learning_rate=0.0002, beta_1=0.5), loss='binary_crossentropy')


# Compile combined model
discriminator.trainable = False
combined_model = models.Sequential([generator, discriminator])
combined_model.compile(optimizer=optimizers.Adam(learning_rate=0.0002, beta_1=0.5), loss='binary_crossentropy')


# Training loop
for epoch in range(NUM_EPOCHS):
    for batch in train_data:
        nitin_images = batch[:NUM_NITIN_IMAGES]
        mountain_images = batch[NUM_NITIN_IMAGES:]

        # Train discriminator
        real_images = batch[:BATCH_SIZE // 2]
        fake_images = generator.predict(np.random.randn(BATCH_SIZE // 2, LATENT_DIM))
        # Use half batch size for real and fake images
        discriminator_loss_real = discriminator.train_on_batch(real_images, np.ones((BATCH_SIZE // 2, 1)))
        discriminator_loss_fake = discriminator.train_on_batch(fake_images, np.zeros((BATCH_SIZE // 2, 1)))
        discriminator_loss = 0.5 * np.add(discriminator_loss_real, discriminator_loss_fake)

        # Train generator
        noise = np.random.randn(BATCH_SIZE, LATENT_DIM)
        generator_loss = combined_model.train_on_batch(noise, np.ones((BATCH_SIZE, 1)))

        print(f"Epoch: {epoch + 1}, Discriminator Loss: {discriminator_loss}, Generator Loss: {generator_loss}")

    # Save model after each epoch
    generator.save(f"gan_model\\generator_model{epoch + 1}.h5")
    discriminator.save(f"gan_model\\discriminator_model{epoch + 1}.h5")
    combined_model.save(f"gan_model\\combined_model{epoch + 1}.h5")


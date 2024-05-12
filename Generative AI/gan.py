import tensorflow as tf
from tensorflow.keras import layers, models, datasets

# Load CIFAR-10 dataset
(train_images, _), (_, _) = datasets.cifar10.load_data()

# Normalize pixel values to range [-1, 1]
train_images = (train_images - 127.5) / 127.5

# Define the generator
def build_generator():
    model = models.Sequential([
        layers.Dense(4 * 4 * 256, use_bias=False, input_shape=(100,)),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Reshape((4, 4, 256)),
        layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2DTranspose(3, (4, 4), strides=(2, 2), padding='same', use_bias=False, activation='tanh')
    ])
    return model

# Define the discriminator
def build_discriminator():
    model = models.Sequential([
        layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[32, 32, 3]),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(1)
    ])
    return model

# Define the GAN
def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = models.Sequential([generator, discriminator])
    return model

# Create instances of generator, discriminator, and GAN
generator = build_generator()
discriminator = build_discriminator()
gan = build_gan(generator, discriminator)

# Compile discriminator
discriminator.compile(optimizer='adam', loss='binary_crossentropy')

# Compile GAN
gan.compile(optimizer='adam', loss='binary_crossentropy')

# Train the GAN
batch_size = 128
epochs = 50
num_batches = train_images.shape[0] // batch_size

for epoch in range(epochs):
    for batch_idx in range(num_batches):
        # Generate random noise
        noise = tf.random.normal([batch_size, 100])

        # Generate fake images from noise
        fake_images = generator.predict(noise)

        # Select a batch of real images
        real_images = train_images[batch_idx * batch_size : (batch_idx + 1) * batch_size]

        # Combine real and fake images
        combined_images = tf.concat([real_images, fake_images], axis=0)

        # Create labels for real and fake images
        real_labels = tf.ones((batch_size, 1))
        fake_labels = tf.zeros((batch_size, 1))
        combined_labels = tf.concat([real_labels, fake_labels], axis=0)

        # Train discriminator
        discriminator_loss = discriminator.train_on_batch(combined_images, combined_labels)

        # Generate new random noise
        noise = tf.random.normal([batch_size, 100])

        # Create misleading labels
        misleading_labels = tf.ones((batch_size, 1))

        # Train generator
        gan_loss = gan.train_on_batch(noise, misleading_labels)

    print(f'Epoch {epoch + 1}/{epochs}, Discriminator Loss: {discriminator_loss}, GAN Loss: {gan_loss}')

# Save the generator model
generator.save('generator_model.h5')

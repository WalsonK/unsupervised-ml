import tensorflow as tf
from keras import layers, Sequential, optimizers
from tqdm import tqdm  # Correction de l'importation de tqdm

class Gan:
    def __init__(self, latent_space_size=10, image_size=784):
        self.latent_space_size = latent_space_size
        self.image_size = image_size

        self.generator = Sequential([
            layers.Input(shape=(latent_space_size,)),
            layers.Dense(128, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(512, activation='relu'),
            layers.Dense(image_size, activation='sigmoid')
        ])
        self.discriminator = Sequential([
            layers.Input(shape=(image_size,)),
            layers.Dense(512, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])

        # Initialisation des optimisateurs
        self.generator.optimizer = optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        self.discriminator.optimizer = optimizers.Adam(learning_rate=0.0002, beta_1=0.5)

    def train(self, X, batch_size, n_epochs):
        for epoch in tqdm(range(n_epochs), desc="Training Epochs"):
            # Get half the batch size real and tag 1
            half_batch = batch_size // 2
            real_images = X[:half_batch]
            real_labels = tf.ones((half_batch, 1))

            # Generate fake images and tag 0
            noise = tf.random.normal((half_batch, self.latent_space_size))
            fake_images = self.generator(noise)
            fake_labels = tf.zeros((half_batch, 1))

            # Concatenate real and fake images
            combined_images = tf.concat([real_images, fake_images], axis=0)
            combined_labels = tf.concat([real_labels, fake_labels], axis=0)

            # Shuffle the combined images and labels
            shuffled_indices = tf.random.shuffle(tf.range(tf.shape(combined_images)[0]))
            combined_images = tf.gather(combined_images, shuffled_indices)
            combined_labels = tf.gather(combined_labels, shuffled_indices)

            # Train the discriminator
            self.discriminator.trainable = True
            with tf.GradientTape() as tape:
                predictions = self.discriminator(combined_images)
                loss = tf.keras.losses.binary_crossentropy(combined_labels, predictions)
            gradients = tape.gradient(loss, self.discriminator.trainable_variables)
            self.discriminator.optimizer.apply_gradients(zip(gradients, self.discriminator.trainable_variables))

            # Fix the discriminator's weights
            self.discriminator.trainable = False

            # Generate fake images and tag 1
            noise = tf.random.normal((batch_size, self.latent_space_size))
            fake_images = self.generator(noise)
            fake_labels = tf.ones((batch_size, 1))

            # Train the generator
            with tf.GradientTape() as tape:
                predictions = self.discriminator(fake_images)
                loss = tf.keras.losses.binary_crossentropy(fake_labels, predictions)
            gradients = tape.gradient(loss, self.generator.trainable_variables)
            self.generator.optimizer.apply_gradients(zip(gradients, self.generator.trainable_variables))


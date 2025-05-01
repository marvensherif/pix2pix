#train.py
from losses import discriminator_loss, generator_loss
from model import discriminator, generator
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import time
import os
import matplotlib.pyplot as plt
from data_generator import data_generator  # Updated generator
from preprocessing import random_jitter    # Add this import

# Load models
gen = generator()
disc = discriminator()

# Set optimizers
generator_optimizer = Adam(learning_rate=2e-4, beta_1=0.5)
discriminator_optimizer = Adam(learning_rate=2e-4, beta_1=0.5)

# Save generated images
def save_images(model, test_input, target, epoch):
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15, 15))
    display_list = [test_input[0], target[0], prediction[0]]
    title = ["Input Image", "Ground Truth", "Prediction Image"]
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        plt.imshow(display_list[i] * 0.5 + 0.5)  # Rescale
        plt.axis("off")
    os.makedirs("output", exist_ok=True)
    plt.savefig(f"output/epoch_{epoch}.jpg")
    plt.close()

# Save model weights
def save_model_weights(epoch):
    os.makedirs('checkpoints', exist_ok=True)
    gen.save_weights(f'checkpoints/gen_weights_epoch_{epoch}.h5')
    disc.save_weights(f'checkpoints/disc_weights_epoch_{epoch}.h5')

@tf.function
def train_step(input_image, target, epoch):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = gen(input_image, training=True)
        disc_real_output = disc([input_image, target], training=True)
        disc_generated_output = disc([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_total_loss, gen.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss, disc.trainable_variables)

        generator_optimizer.apply_gradients(zip(generator_gradients, gen.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(discriminator_gradients, disc.trainable_variables))

        return gen_total_loss, disc_loss

def preprocess_with_jitter(input_img, target_img):
    return random_jitter(input_img, target_img)

def get_tf_dataset(generator_func, batch_size=1, buffer_size=100):
    output_signature = (
        tf.TensorSpec(shape=(256, 256, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(256, 256, 3), dtype=tf.float32),
    )
    ds = tf.data.Dataset.from_generator(generator_func, output_signature=output_signature)
    return ds.map(preprocess_with_jitter, num_parallel_calls=tf.data.AUTOTUNE)\
             .batch(batch_size)\
             .shuffle(buffer_size)\
             .prefetch(tf.data.AUTOTUNE)

def fit(train_ds, epochs, test_ds):
    for epoch in range(epochs):
        start = time.time()

        for input_, target in test_ds.take(1):
            save_images(gen, input_, target, epoch)

        print(f"Epoch {epoch}")
        for n, (input_, target) in train_ds.enumerate():
            gen_loss, disc_loss = train_step(input_, target, epoch)

        print(f"Generator loss: {gen_loss:.2f} | Discriminator loss: {disc_loss:.2f}")
        print(f"Time taken for epoch {epoch+1}: {time.time() - start:.2f} sec\n")

        save_model_weights(epoch)

# Start training
epochs = 50
train_dataset = get_tf_dataset(data_generator)
test_dataset = get_tf_dataset(data_generator)
fit(train_dataset, epochs, test_dataset)

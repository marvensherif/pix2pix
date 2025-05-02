import os
import time
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from model import generator, discriminator
from losses import generator_loss, discriminator_loss
from data_generator import data_generator
from preprocessing import random_jitter
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import pandas as pd

# === Setup Logging ===
LOG_PATH = "training_log.xlsx"
def log_to_excel(history):
    # Save the history to an Excel file
    df = pd.DataFrame(history)
    df.to_excel(LOG_PATH, index=False)
    print(f"Logs saved to {LOG_PATH}")

# === GPU Configuration ===
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    tf.config.optimizer.set_experimental_options({'layout_optimizer': False})

# === Models ===
gen = generator()
disc = discriminator()

# === Optimizers & Scheduler ===
generator_optimizer = Adam(learning_rate=2e-4, beta_1=0.5)
discriminator_optimizer = Adam(learning_rate=2e-4, beta_1=0.5)
lr_scheduler = ReduceLROnPlateau(monitor='val_gen_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

# === Save model weights ===
def save_model_weights(epoch):
    os.makedirs('checkpoints', exist_ok=True)
    gen.save_weights(f'checkpoints/gen_weights_epoch_{epoch}.weights.h5')
    disc.save_weights(f'checkpoints/disc_weights_epoch_{epoch}.weights.h5')

def save_final_weights():
    run_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    gen.save_weights(f'checkpoints/gen_weights_final_{run_time}.weights.h5')
    disc.save_weights(f'checkpoints/disc_weights_final_{run_time}.weights.h5')

# === Visual Output ===
def save_images(model, test_input, target, epoch):
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15, 15))
    display_list = [test_input[0], target[0], prediction[0]]
    title = ["Input Image", "Ground Truth", "Prediction"]
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(title[i])
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis("off")
    os.makedirs("output", exist_ok=True)
    plt.savefig(f"output/epoch_{epoch}.jpg")
    plt.close()

# === Dataset Loader ===
def get_tf_dataset(generator_func, batch_size=1, buffer_size=100):
    output_signature = (
        tf.TensorSpec(shape=(256, 256, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(256, 256, 3), dtype=tf.float32),
    )
    ds = tf.data.Dataset.from_generator(generator_func, output_signature=output_signature)
    ds = ds.map(random_jitter, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# === Early Stopping Class ===
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_loss = np.inf
        self.epochs_without_improvement = 0

    def check_early_stop(self, val_loss):
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
        
        return self.epochs_without_improvement >= self.patience

# === Training Step ===
def train_step(input_image, target):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = gen(input_image, training=True)
        disc_real_output = disc([input_image, target], training=True)
        disc_generated_output = disc([input_image, gen_output], training=True)

        gen_total_loss, _, _ = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_total_loss, gen.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, disc.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients, gen.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, disc.trainable_variables))

    gen_grad_norm = tf.linalg.global_norm(generator_gradients).numpy()
    disc_grad_norm = tf.linalg.global_norm(discriminator_gradients).numpy()
    return gen_total_loss, disc_loss, gen_grad_norm, disc_grad_norm

# === Training Loop ===
def fit(train_ds, val_ds, epochs, steps_per_epoch):
    history = {"epoch": [], "gen_loss": [], "disc_loss": [], "val_gen_loss": [], "val_disc_loss": [], "gen_grad": [], "disc_grad": []}
    early_stopping = EarlyStopping(patience=10, min_delta=0.0)

    for epoch in range(epochs):
        start = time.time()

        # Visual check
        for input_, target in val_ds.take(1):
            save_images(gen, input_, target, epoch)

        gen_loss_avg, disc_loss_avg = 0, 0
        gen_grad_total, disc_grad_total = 0, 0

        for n, (input_, target) in train_ds.enumerate():
            if n >= steps_per_epoch:
                break
            g_loss, d_loss, g_grad, d_grad = train_step(input_, target)
            gen_loss_avg += g_loss.numpy()
            disc_loss_avg += d_loss.numpy()
            gen_grad_total += g_grad
            disc_grad_total += d_grad

        # Validation loss
        val_input, val_target = next(iter(val_ds))
        val_output = gen(val_input, training=False)
        val_disc_real = disc([val_input, val_target], training=False)
        val_disc_fake = disc([val_input, val_output], training=False)
        val_gen_total_loss, _, _ = generator_loss(val_disc_fake, val_output, val_target)
        val_disc_loss = discriminator_loss(val_disc_real, val_disc_fake)

        # Track history
        history["epoch"].append(epoch + 1)
        history["gen_loss"].append(gen_loss_avg / steps_per_epoch)
        history["disc_loss"].append(disc_loss_avg / steps_per_epoch)
        history["val_gen_loss"].append(val_gen_total_loss.numpy())
        history["val_disc_loss"].append(val_disc_loss.numpy())
        history["gen_grad"].append(gen_grad_total / steps_per_epoch)
        history["disc_grad"].append(disc_grad_total / steps_per_epoch)

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train => Gen Loss: {gen_loss_avg/steps_per_epoch:.4f}, Disc Loss: {disc_loss_avg/steps_per_epoch:.4f}")
        print(f"Val   => Gen Loss: {val_gen_total_loss.numpy():.4f}, Disc Loss: {val_disc_loss.numpy():.4f}")
        print(f"Gradient Norms => Gen: {gen_grad_total/steps_per_epoch:.4f}, Disc: {disc_grad_total/steps_per_epoch:.4f}")
        print(f"Time: {time.time() - start:.2f}s")

        # Early stopping check
        if early_stopping.check_early_stop(val_gen_total_loss.numpy()):
            print(f"Early stopping triggered after epoch {epoch + 1}")
            break

        lr_scheduler.on_epoch_end(epoch, logs={"val_gen_loss": val_gen_total_loss})
        save_model_weights(epoch)

    save_final_weights()

    # Log history to Excel
    log_to_excel(history)
    return history

# === Training Runner ===
epochs = 10
steps_per_epoch = 100
train_dataset = get_tf_dataset(data_generator)
val_dataset = get_tf_dataset(data_generator)
history = fit(train_dataset, val_dataset, epochs, steps_per_epoch)

# === Plotting Learning Curves ===
def plot_learning_curves(history):
    os.makedirs("plots", exist_ok=True)
    plt.plot(history['gen_loss'], label='Train Generator Loss')
    plt.plot(history['val_gen_loss'], label='Val Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("plots/generator_loss_curve.png")
    plt.close()

    plt.plot(history['disc_loss'], label='Train Discriminator Loss')
    plt.plot(history['val_disc_loss'], label='Val Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("plots/discriminator_loss_curve.png")
    plt.close()

    plt.plot(history['gen_grad'], label='Gen Grad Norm')
    plt.plot(history['disc_grad'], label='Disc Grad Norm')
    plt.xlabel('Epoch')
    plt.ylabel('Gradient Norm')
    plt.legend()
    plt.savefig("plots/gradient_norms.png")
    plt.close()

plot_learning_curves(history)

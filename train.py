import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from model import generator, discriminator
from losses import generator_loss, discriminator_loss
from data_generator import data_generator
from preprocessing import random_jitter
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate

# === Configuration ===
LOG_DIR = "logs"
PLOT_DIR = "plots"
CHECKPOINT_DIR = "checkpoints"
IMG_SIZE = 256
BATCH_SIZE = 1
BUFFER_SIZE = 1000
LOG_FILENAME = "training_log.txt"
MODEL_SUMMARY_FILENAME = "model_summaries.txt"

# === Setup Directories ===
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# === Logging ===
def log_to_txt_table(history):
    df = pd.DataFrame(history)
    log_path = os.path.join(LOG_DIR, LOG_FILENAME)
    with open(log_path, "w") as f:
        f.write(tabulate(df, headers='keys', tablefmt='pretty'))
    print(f"Logs saved to {log_path}")

# === GPU Configuration ===
def configure_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            print(f"GPU config error: {e}")

configure_gpu()

# === Model Initialization ===
gen = generator()
disc = discriminator()

# === Optimizers ===
generator_optimizer = Adam(2e-4, beta_1=0.5)
discriminator_optimizer = Adam(2e-4, beta_1=0.5)

# === Model Summary Saving ===
def save_model_summaries():
    """Save model architectures to text files"""
    summary_path = os.path.join(LOG_DIR, MODEL_SUMMARY_FILENAME)
    with open(summary_path, "w") as f:
        f.write("=== Generator Architecture ===\n")
        gen.summary(print_fn=lambda x: f.write(x + "\n"))
        f.write("\n=== Discriminator Architecture ===\n")
        disc.summary(print_fn=lambda x: f.write(x + "\n"))
    print(f"Model summaries saved to {summary_path}")

# === Optimizer Initialization ===
def initialize_optimizers():
    """Build optimizer variables using dummy data"""
    dummy_input = tf.random.normal([1, IMG_SIZE, IMG_SIZE, 3])
    
    # Initialize generator optimizer
    with tf.GradientTape() as tape:
        gen_output = gen(dummy_input, training=True)
        loss = tf.reduce_mean(gen_output)
    grads = tape.gradient(loss, gen.trainable_variables)
    generator_optimizer.apply_gradients(zip(grads, gen.trainable_variables))
    
    # Initialize discriminator optimizer
    with tf.GradientTape() as tape:
        disc_output = disc([dummy_input, gen_output], training=True)
        loss = tf.reduce_mean(disc_output)
    grads = tape.gradient(loss, disc.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(grads, disc.trainable_variables))

initialize_optimizers()

# === Checkpoint Handling ===
def validate_checkpoint(epoch):
    """Verify all required checkpoint files exist"""
    required_files = [
        f"gen_epoch_{epoch}.weights.h5",
        f"disc_epoch_{epoch}.weights.h5",
        f"gen_optimizer_epoch_{epoch}.npz",
        f"disc_optimizer_epoch_{epoch}.npz"
    ]
    missing = [f for f in required_files if not os.path.exists(os.path.join(CHECKPOINT_DIR, f))]
    if missing:
        raise FileNotFoundError(f"Missing checkpoint files: {missing}")
    return True

def save_checkpoint(epoch):
    """Save complete training state with actual epoch number"""
    # Save models
    gen.save_weights(os.path.join(CHECKPOINT_DIR, f"gen_epoch_{epoch}.weights.h5"))
    disc.save_weights(os.path.join(CHECKPOINT_DIR, f"disc_epoch_{epoch}.weights.h5"))
    
    # Save optimizer states
    def save_optimizer(optimizer, prefix):
        variables = optimizer.variables
        state = {f"{i:04d}": v.numpy() for i, v in enumerate(variables)}
        np.savez(os.path.join(CHECKPOINT_DIR, f"{prefix}_epoch_{epoch}.npz"), **state)
    
    save_optimizer(generator_optimizer, "gen_optimizer")
    save_optimizer(discriminator_optimizer, "disc_optimizer")
    print(f"Saved checkpoint for epoch {epoch}")

def load_checkpoint(epoch):
    """Load checkpoint with strict validation"""
    validate_checkpoint(epoch)
    
    # Load models
    gen.load_weights(os.path.join(CHECKPOINT_DIR, f"gen_epoch_{epoch}.weights.h5"))
    disc.load_weights(os.path.join(CHECKPOINT_DIR, f"disc_epoch_{epoch}.weights.h5"))
    
    # Load optimizers
    def load_optimizer(optimizer, prefix):
        path = os.path.join(CHECKPOINT_DIR, f"{prefix}_epoch_{epoch}.npz")
        data = np.load(path, allow_pickle=True)
        weights = [data[f"{i:04d}"] for i in range(len(data.files))]
        optimizer.set_weights(weights)
    
    load_optimizer(generator_optimizer, "gen_optimizer")
    load_optimizer(discriminator_optimizer, "disc_optimizer")
    print(f"Loaded checkpoint from epoch {epoch}")

# === Training Visualization ===
def save_images(model, test_input, target, epoch):
    prediction = model(test_input, training=False)
    plt.figure(figsize=(10, 10))
    
    display_list = [test_input[0], target[0], prediction[0]]
    titles = ['Input', 'Ground Truth', 'Prediction']
    
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(titles[i])
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    
    plot_path = os.path.join(PLOT_DIR, f"epoch_{epoch}.png")
    plt.savefig(plot_path)
    plt.close()

# === Data Pipeline ===
def get_dataset(generator_func):
    output_signature = (
        tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32)
    )
    
    ds = tf.data.Dataset.from_generator(
        generator_func,
        output_signature=output_signature
    )
    
    ds = ds.map(random_jitter, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(BUFFER_SIZE)
    ds = ds.batch(BATCH_SIZE)
    return ds.prefetch(tf.data.AUTOTUNE)

# === Training Core ===
class TrainingState:
    def __init__(self, patience=10, min_delta=0.01):
        self.best_loss = float('inf')
        self.patience = patience
        self.min_delta = min_delta
        self.wait = 0

    def check_early_stop(self, current_loss):
        if (self.best_loss - current_loss) > self.min_delta:
            self.best_loss = current_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                return True
        return False

def train_step(input_image, target):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = gen(input_image, training=True)
        
        disc_real = disc([input_image, target], training=True)
        disc_fake = disc([input_image, gen_output], training=True)
        
        gen_loss_total, _, _ = generator_loss(disc_fake, gen_output, target)
        disc_loss = discriminator_loss(disc_real, disc_fake)
    
    gen_gradients = gen_tape.gradient(gen_loss_total, gen.trainable_variables)
    disc_gradients = disc_tape.gradient(disc_loss, disc.trainable_variables)
    
    gen_grad_norm = tf.linalg.global_norm(gen_gradients)
    disc_grad_norm = tf.linalg.global_norm(disc_gradients)
    
    generator_optimizer.apply_gradients(zip(gen_gradients, gen.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(disc_gradients, disc.trainable_variables))
    
    return gen_loss_total, disc_loss, gen_grad_norm, disc_grad_norm

def validate(val_ds):
    val_input, val_target = next(iter(val_ds))
    val_gen_output = gen(val_input, training=False)
    val_disc_real = disc([val_input, val_target], training=False)
    val_disc_fake = disc([val_input, val_gen_output], training=False)
    val_gen_loss = generator_loss(val_disc_fake, val_gen_output, val_target)[0]
    val_disc_loss = discriminator_loss(val_disc_real, val_disc_fake)
    return val_input, val_target, val_gen_loss.numpy(), val_disc_loss.numpy()

# === Training Loop ===
def fit(train_ds, val_ds, total_epochs, steps_per_epoch, initial_epoch=0):
    history = {
        'epoch': [], 'gen_loss': [], 'disc_loss': [],
        'val_gen_loss': [], 'val_disc_loss': [],
        'gen_grad_norm': [], 'disc_grad_norm': [],
        'lr': []
    }
    
    early_stop = TrainingState()
    best_val_loss = float('inf')
    current_lr = float(generator_optimizer.learning_rate.numpy())
    min_lr = 1e-6
    patience_counter = 0
    lr_patience = 3

    if initial_epoch > 0:
        load_checkpoint(initial_epoch - 1)
        print(f"Resuming training from epoch {initial_epoch}")

    for epoch in range(initial_epoch, total_epochs):
        start_time = time.time()
        gen_losses, disc_losses = [], []
        gen_grad_norms, disc_grad_norms = [], []
        
        # Training phase
        for step, (input_image, target) in enumerate(train_ds.take(steps_per_epoch)):
            gen_loss, disc_loss, gen_grad, disc_grad = train_step(input_image, target)
            gen_losses.append(gen_loss.numpy())
            disc_losses.append(disc_loss.numpy())
            gen_grad_norms.append(gen_grad.numpy())
            disc_grad_norms.append(disc_grad.numpy())
            
            if step % 10 == 0:
                print(f"Epoch {epoch+1} | Step {step+1}/{steps_per_epoch} | "
                      f"Gen Loss: {gen_loss:.4f} | Disc Loss: {disc_loss:.4f} | "
                      f"Grad Norms: G={gen_grad:.2f}, D={disc_grad:.2f}")

        # Validation
        val_input, val_target, val_gen_loss, val_disc_loss = validate(val_ds)
        
        # Learning rate adjustment
        if val_gen_loss < best_val_loss - 0.01:
            best_val_loss = val_gen_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= lr_patience:
                current_lr = max(current_lr * 0.5, min_lr)
                generator_optimizer.learning_rate.assign(current_lr)
                discriminator_optimizer.learning_rate.assign(current_lr)
                print(f"Reduced learning rate to {current_lr:.2e}")
                patience_counter = 0

        # Save history
        history['epoch'].append(epoch+1)
        history['gen_loss'].append(np.mean(gen_losses))
        history['disc_loss'].append(np.mean(disc_losses))
        history['val_gen_loss'].append(val_gen_loss)
        history['val_disc_loss'].append(val_disc_loss)
        history['gen_grad_norm'].append(np.mean(gen_grad_norms))
        history['disc_grad_norm'].append(np.mean(disc_grad_norms))
        history['lr'].append(current_lr)
        
        # Save checkpoint and visuals
        save_checkpoint(epoch)
        save_images(gen, val_input, val_target, epoch+1)

        # Early stopping
        if early_stop.check_early_stop(val_gen_loss):
            print(f"Early stopping at epoch {epoch+1}")
            break

        # Epoch summary
        print(f"\nEpoch {epoch+1}/{total_epochs} ({time.time()-start_time:.1f}s)")
        print(f"Train Loss: G={np.mean(gen_losses):.4f}, D={np.mean(disc_losses):.4f}")
        print(f"Val Loss:   G={val_gen_loss:.4f}, D={val_disc_loss:.4f}")
        print(f"Grad Norms: G={np.mean(gen_grad_norms):.2f}, D={np.mean(disc_grad_norms):.2f}")
        print(f"Learning Rate: {current_lr:.2e}\n")
    
    # Final saves
    gen.save_weights(os.path.join(CHECKPOINT_DIR, "gen_final.weights.h5"))
    disc.save_weights(os.path.join(CHECKPOINT_DIR, "disc_final.weights.h5"))
    log_to_txt_table(history)
    
    # Save plots
    plt.figure(figsize=(12, 8))
    plt.plot(history['gen_loss'], label='Generator Train')
    plt.plot(history['val_gen_loss'], label='Generator Val')
    plt.plot(history['disc_loss'], label='Discriminator Train')
    plt.plot(history['val_disc_loss'], label='Discriminator Val')
    plt.title("Training Progress")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(PLOT_DIR, 'training_curves.png'))
    plt.close()
    
    plt.figure(figsize=(12, 6))
    plt.plot(history['gen_grad_norm'], label='Generator Grad Norm')
    plt.plot(history['disc_grad_norm'], label='Discriminator Grad Norm')
    plt.title("Gradient Norms During Training")
    plt.xlabel("Epoch")
    plt.ylabel("Gradient Norm")
    plt.legend()
    plt.savefig(os.path.join(PLOT_DIR, 'gradient_norms.png'))
    plt.close()
    
    return history

# === Main Execution ===
if __name__ == "__main__":
    save_model_summaries()
    
    # Training parameters
    total_epochs = 7
    steps_per_epoch = 5
    initial_epoch = 2
    
    # Initialize datasets
    train_dataset = get_dataset(data_generator).repeat()
    val_dataset = get_dataset(data_generator).repeat()
    
    # Start training
    history = fit(
        train_ds=train_dataset,
        val_ds=val_dataset,
        total_epochs=total_epochs,
        steps_per_epoch=steps_per_epoch,
        initial_epoch=initial_epoch
    )
    
    print("Training completed successfully!")
#test.py
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from model import generator  # Your generator model function
from data_generator import data_generator  # Updated generator that handles resize and normalize

# Ensure output directory exists
os.makedirs("output", exist_ok=True)

# Load trained generator model
gen = generator()
gen.load_weights('checkpoints/gen_weights_final.h5')  # Make sure this file exists

# Save test result images
def save_images(model, test_input, target, epoch):
    prediction = model(test_input, training=False)

    plt.figure(figsize=(15, 15))
    display_list = [test_input[0], target[0], prediction[0]]
    title = ["Input Image", "Ground Truth", "Prediction Image"]

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(title[i])
        plt.imshow(display_list[i] * 0.5 + 0.5)  # Rescale to [0, 1]
        plt.axis("off")

    plt.savefig(f"output/test_epoch_{epoch}.jpg")
    plt.close()

# Create test dataset (no random_jitter, just normalized/resized)
def get_tf_dataset(generator_func, batch_size=1):
    output_signature = (
        tf.TensorSpec(shape=(256, 256, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(256, 256, 3), dtype=tf.float32),
    )
    ds = tf.data.Dataset.from_generator(generator_func, output_signature=output_signature)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Test loop
def test(test_ds, epochs):
    for epoch in range(epochs):
        print(f"Testing Epoch {epoch}")
        for input_, target in test_ds.take(1):  # Just one sample for visual inspection
            save_images(gen, input_, target, epoch)

# Set up and run test
test_dataset = get_tf_dataset(data_generator)
test(test_dataset, epochs=50)

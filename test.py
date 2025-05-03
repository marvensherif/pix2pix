import os
import argparse
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from model import generator
from data_generator import data_generator

def parse_args():
    parser = argparse.ArgumentParser(description='Test pix2pix model')
    parser.add_argument('--weights_path', type=str, required=True,
                      help='Path to generator weights file')
    parser.add_argument('--output_dir', type=str, default='test_output',
                      help='Directory to save test results')
    parser.add_argument('--num_samples', type=int, default=10,
                      help='Number of samples to test (for generator mode)')
    parser.add_argument('--data_source', choices=['generator', 'custom'], default='generator',
                      help='Data source: generator or custom images')
    parser.add_argument('--input_dir', type=str,
                      help='Directory containing custom input images')
    return parser.parse_args()

def preprocess_image(image_path, target_size=(256, 256)):
    """Load and preprocess custom image"""
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3)
    image = tf.image.resize(image, target_size)
    image = (image / 127.5) - 1.0  # Normalize to [-1, 1]
    return image

def save_images(model, test_input, target, save_path):
    """Save comparison image with optional target"""
    prediction = model(test_input, training=False)
    
    plt.figure(figsize=(15, 5))
    titles = ["Input Image", "Prediction"]
    display_list = [test_input[0], prediction[0]]
    
    if target is not None:
        titles.append("Ground Truth")
        display_list.append(target[0])
    
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(titles[i])
        plt.imshow(display_list[i] * 0.5 + 0.5)  # Scale from [-1,1] to [0,1]
        plt.axis("off")
    
    plt.savefig(save_path)
    plt.close()

def test(weights_path, output_dir, num_samples, data_source, input_dir=None):
    # Load generator model
    gen = generator()
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found at {weights_path}")
        
    gen.load_weights(weights_path)
    print(f"Successfully loaded weights from {weights_path}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    if data_source == 'generator':
        # Create finite test dataset
        test_dataset = tf.data.Dataset.from_generator(
            data_generator,
            output_signature=(
                tf.TensorSpec(shape=(256, 256, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(256, 256, 3), dtype=tf.float32)
            )
        ).take(num_samples).batch(1).prefetch(tf.data.AUTOTUNE)

        # Test with generator samples
        for i, (input_img, target_img) in enumerate(test_dataset):
            save_path = os.path.join(output_dir, f'test_result_{i:04d}.png')
            save_images(gen, input_img, target_img, save_path)
            print(f"Saved test result {i+1} to {save_path}")

    elif data_source == 'custom':
        if not input_dir:
            raise ValueError("Input directory required for custom data source")
        
        # Process custom images
        image_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
                      if f.lower().endswith(('png', 'jpg', 'jpeg'))]
        
        for i, image_path in enumerate(image_paths[:num_samples]):
            input_img = preprocess_image(image_path)
            input_img = tf.expand_dims(input_img, axis=0)  # Add batch dimension
            
            save_path = os.path.join(output_dir, f'custom_result_{i:04d}.png')
            save_images(gen, input_img, None, save_path)  # No target available
            print(f"Processed custom image {i+1} to {save_path}")

if __name__ == "__main__":
    args = parse_args()
    test(
        args.weights_path,
        args.output_dir,
        args.num_samples,
        args.data_source,
        args.input_dir
    )

  #python test.py --weights_path checkpoints/gen.weights.h5 --data_source generator --num_samples 5
  #python test.py --weights_path checkpoints/gen.weights.h5 --data_source custom --input_dir my_images --num_samples 20








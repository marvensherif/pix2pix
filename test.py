# test.py
import os
import argparse
import tensorflow as tf
import matplotlib.pyplot as plt
from model import generator
from data_generator import data_generator

def parse_args():
    parser = argparse.ArgumentParser(description='Test pix2pix model')
    parser.add_argument('--weights_path', type=str, required=True,
                       help='Path to generator weights file')
    parser.add_argument('--output_dir', type=str, default='test_output',
                       help='Directory to save test results')
    return parser.parse_args()

def save_images(model, test_input, target, save_path):
    prediction = model(test_input, training=False)
    
    plt.figure(figsize=(15, 15))
    display_list = [test_input[0], target[0], prediction[0]]
    titles = ["Input Image", "Ground Truth", "Prediction"]
    
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(titles[i])
        plt.imshow(display_list[i] * 0.5 + 0.5)  # Scale from [-1,1] to [0,1]
        plt.axis("off")
    
    plt.savefig(save_path)
    plt.close()

def test(weights_path, output_dir):
    # Create generator and load weights
    gen = generator()
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found at {weights_path}")
        
    gen.load_weights(weights_path)
    print(f"Successfully loaded weights from {weights_path}")

       # Create dataset (without augmentation)
    test_dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_signature=(
            tf.TensorSpec(shape=(256, 256, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(256, 256, 3), dtype=tf.float32)
        )
    ).batch(1).prefetch(tf.data.AUTOTUNE)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Run test
    for i, (input_img, target_img) in enumerate(test_dataset):
        save_path = os.path.join(output_dir, f'test_result_{i:04d}.png')
        save_images(gen, input_img, target_img, save_path)
        print(f"Saved test result {i+1} to {save_path}")

if __name__ == "__main__":
    args = parse_args()
    test(args.weights_path, args.output_dir)

#python test.py --weights_path checkpoints/gen_epoch_100.weights.h5 --output_dir my_test_results
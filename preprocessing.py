# preprocessing.py

import tensorflow as tf

IMAGE_SIZE = 256  # Image size after resize

def normalize(input_image, real_image):
    """Normalize images to [-1, 1] range"""
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1
    return input_image, real_image

def resize(input_image, real_image):
    """Resize images to (IMAGE_SIZE, IMAGE_SIZE)"""
    input_image = tf.image.resize(input_image, [IMAGE_SIZE, IMAGE_SIZE], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [IMAGE_SIZE, IMAGE_SIZE], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return input_image, real_image

def random_jitter(input_image, real_image):
    # Apply horizontal flip to both (to maintain alignment)
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)

    # You can also add random cropping/resizing here for both if needed

    # Apply brightness to input only
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.random_brightness(input_image, max_delta=0.2)

    # Apply contrast to input only
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.random_contrast(input_image, lower=0.8, upper=1.2)

    # Apply saturation to input only
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.random_saturation(input_image, lower=0.8, upper=1.2)

    # Apply hue shift to input only
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.random_hue(input_image, max_delta=0.05)

    # Add Gaussian noise to input only
    if tf.random.uniform(()) > 0.5:
        noise = tf.random.normal(shape=tf.shape(input_image), mean=0.0, stddev=0.05)
        input_image = tf.clip_by_value(input_image + noise, 0.0, 1.0)

    return input_image, real_image




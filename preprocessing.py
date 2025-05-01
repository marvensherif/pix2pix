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
    """Random horizontal flip (applied unconditionally)"""
    input_image = tf.image.flip_left_right(input_image)
    real_image = tf.image.flip_left_right(real_image)
    return input_image, real_image

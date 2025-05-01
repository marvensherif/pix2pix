# data_generator.py

import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import tensorflow as tf
from preprocessing import resize, normalize  # Import your preprocessing functions

# Configuration
IMAGE_SIZE = (256, 256)

def generate_text():
    chars = "ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnpqrstuvwxyz23456789"
    return ''.join(random.choices(chars, k=random.randint(4, 6)))

def get_random_font(size):
    return ImageFont.load_default().font_variant(size=size)

def create_clean_image():
    bg_color = (255, 255, 255)
    img = Image.new("RGB", IMAGE_SIZE, bg_color)
    draw = ImageDraw.Draw(img)
    text = generate_text()
    max_x = IMAGE_SIZE[0] // 2
    max_y = IMAGE_SIZE[1] // 2
    x = random.randint(20, max_x)
    y = random.randint(20, max_y)

    for char in text:
        font_size = random.randint(32, 48)
        font = get_random_font(font_size)
        bbox = font.getbbox(char)
        char_width = bbox[2] - bbox[0]
        char_height = bbox[3] - bbox[1]
        x = max(0, min(x, IMAGE_SIZE[0] - char_width - 5))
        y = max(0, min(y, IMAGE_SIZE[1] - char_height - 5))
        draw.text(
            (x + random.randint(-2, 2), y + random.randint(-2, 2)),
            char,
            fill=(0, 0, 0),
            font=font
        )
        x += char_width + random.randint(1, 5)
        y += random.randint(-3, 3)
        x = min(x, IMAGE_SIZE[0] - 20)
        y = min(max(y, 20), IMAGE_SIZE[1] - 20)

    return img, text

def apply_noise(image):
    arr = np.array(image.convert('RGB'))
    h, w = arr.shape[:2]

    if random.random() < 0.8:
        new_bg = np.random.randint(180, 250, size=3)
        mask = (arr > 240).all(axis=2)
        for c in range(3):
            arr[:, :, c][mask] = new_bg[c]

    if random.random() < 0.85:
        num = int(h * w * random.uniform(0.01, 0.08))
        ys, xs = np.random.randint(0, h, num), np.random.randint(0, w, num)
        for y, x in zip(ys, xs):
            if random.random() < 0.92:
                arr[y, x] = random.choice([(0, 0, 0), (255, 255, 255)])
            else:
                arr[y, x] = np.random.randint(0, 255, size=3)

    if random.random() < 0.75:
        for _ in range(random.randint(50, 250)):
            x, y = random.randint(0, w - 1), random.randint(0, h - 1)
            cv2.circle(arr, (x, y), random.randint(1, 2), (0, 0, 0), -1)

    if random.random() < 0.25:
        for _ in range(random.randint(10, 60)):
            color = np.random.randint(0, 255, size=3)
            x, y = random.randint(0, w - 1), random.randint(0, h - 1)
            cv2.circle(arr, (x, y), random.randint(1, 2), color.tolist(), -1)

    if random.random() < 0.6:
        for _ in range(random.randint(3, 8)):
            style = random.choice(["horizontal", "vertical", "diagonal"])
            color = (0, 0, 0) if random.random() < 0.88 else np.random.randint(0, 255, size=3).tolist()
            thickness = random.choice([1, 1, 2])
            if style == "horizontal":
                y = random.randint(0, h - 1)
                cv2.line(arr, (0, y), (w - 1, y), color, thickness)
            elif style == "vertical":
                x = random.randint(0, w - 1)
                cv2.line(arr, (x, 0), (x, h - 1), color, thickness)
            else:
                x1, y1 = random.randint(0, w // 2), random.randint(0, h // 2)
                x2, y2 = random.randint(w // 2, w - 1), random.randint(h // 2, h - 1)
                cv2.line(arr, (x1, y1), (x2, y2), color, thickness)

    if random.random() < 0.95:
        k = random.choice([7, 9, 11, 13])
        arr = cv2.GaussianBlur(arr, (k, k), 0)

    return Image.fromarray(arr)

def data_generator():
    """Yields preprocessed (noisy_image_tensor, clean_image_tensor) pairs"""
    while True:
        clean_img, _ = create_clean_image()
        noisy_img = apply_noise(clean_img)

        # Convert to TensorFlow tensors
        clean_tensor = tf.convert_to_tensor(np.array(clean_img), dtype=tf.float32)
        noisy_tensor = tf.convert_to_tensor(np.array(noisy_img), dtype=tf.float32)

        # Apply resize and normalize
        noisy_tensor, clean_tensor = resize(noisy_tensor, clean_tensor)
        noisy_tensor, clean_tensor = normalize(noisy_tensor, clean_tensor)

        yield noisy_tensor, clean_tensor

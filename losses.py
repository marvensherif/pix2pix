import tensorflow as tf
from keras.losses import BinaryCrossentropy
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model

# Constants (Updated weights)
LAMBDA_L1 = 100          # Kept high for pix2pix-like tasks
LAMBDA_PERCEPTUAL = 0.1  # Reduced due to proper normalization

# Binary crossentropy for GAN
loss_function = BinaryCrossentropy(from_logits=True)

# VGG19 Feature Extractor (Frozen)
vgg = VGG19(include_top=False, weights='imagenet', input_shape=(256, 256, 3))
vgg.trainable = False  # Crucial: freeze VGG weights
vgg_model = Model(
    inputs=vgg.input, 
    outputs=vgg.get_layer('block3_conv3').output  # Common choice for perceptual loss
)

def preprocess_vgg(img):
    """Correct preprocessing for VGG19:
    1. Scale from [-1, 1] to [0, 255] range
    2. Apply VGG-specific mean subtraction
    """
    # Scale from [-1, 1] to [0, 255] in one step
    img = (img + 1.0) * 127.5  
    # Apply VGG preprocessing (BGR mean subtraction)
    return tf.keras.applications.vgg19.preprocess_input(img)

def perceptual_loss(y_true, y_pred):
    """Normalized perceptual loss with correct preprocessing"""
    # Process images through VGG
    y_true_vgg = vgg_model(preprocess_vgg(y_true))
    y_pred_vgg = vgg_model(preprocess_vgg(y_pred))
    
    # Calculate normalized MSE
    mse = tf.reduce_mean(tf.square(y_true_vgg - y_pred_vgg))
    
    # Additional normalization by feature map size
    num_elements = tf.cast(tf.size(y_true_vgg), tf.float32)
    return mse / num_elements

def generator_loss(disc_generated_output, gen_output, target):
    """Combined loss with proper scaling"""
    # GAN loss component
    gan_loss = loss_function(tf.ones_like(disc_generated_output), disc_generated_output)
    
    # L1 loss (assumes images in [-1, 1] range)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    
    # Perceptual loss (now properly scaled)
    perc_loss = perceptual_loss(target, gen_output)
    
    # Weighted sum (note: LAMBDA_PERCEPTUAL is now 0.1)
    total_gen_loss = (
        gan_loss + 
        (LAMBDA_L1 * l1_loss) + 
        (LAMBDA_PERCEPTUAL * perc_loss)
    )
    
    return total_gen_loss, gan_loss, l1_loss, perc_loss

def discriminator_loss(disc_real_output, disc_generated_output):
    """Standard GAN discriminator loss"""
    real_loss = loss_function(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_function(tf.zeros_like(disc_generated_output), disc_generated_output)
    return real_loss + generated_loss
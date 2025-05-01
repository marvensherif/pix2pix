# losses.py

import tensorflow as tf
from keras.losses import BinaryCrossentropy

# Constants
LAMBDA = 100

# Loss function for GANs
loss_function = BinaryCrossentropy(from_logits=True)

# Generator loss
def generator_loss(disc_generated_output, gen_output, target):
    # GAN loss
    gan_loss = loss_function(tf.ones_like(disc_generated_output), disc_generated_output)
    
    # L1 loss
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    
    # Total generator loss
    total_gen_loss = gan_loss + (LAMBDA * l1_loss)
    
    return total_gen_loss, gan_loss, l1_loss

# Discriminator loss
def discriminator_loss(disc_real_output, disc_generated_output):
    # Real loss
    real_loss = loss_function(tf.ones_like(disc_real_output), disc_real_output)
    
    # Generated loss
    generated_loss = loss_function(tf.zeros_like(disc_generated_output), disc_generated_output)
    
    # Total discriminator loss
    total_disc_loss = real_loss + generated_loss
    
    return total_disc_loss

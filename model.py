# model.py

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, LeakyReLU, ReLU, Dropout, Input, Concatenate, ZeroPadding2D
from tensorflow.keras.models import Model, Sequential

# Set default image size (can be overridden externally)
IMAGE_SIZE = 256

# Downsample block
def downsample(filters, size, batchnorm=True):
    init = tf.random_normal_initializer(0., 0.02)
    result = Sequential()
    result.add(Conv2D(filters, size, strides=2, padding="same", kernel_initializer=init, use_bias=False))
    if batchnorm:
        result.add(BatchNormalization())
    result.add(LeakyReLU())
    return result

def upsample(filters, size, dropout=False):
    result = Sequential()
    result.add(Conv2DTranspose(filters, size, strides=2, 
              padding="same", use_bias=False))
    result.add(BatchNormalization())
    if dropout:
        # Add explicit noise_shape and name
       result.add(Dropout(0.5))

    result.add(ReLU())
    return result

# Generator model
def generator():
    inputs = Input(shape=[IMAGE_SIZE, IMAGE_SIZE, 3])
    
    down_stack = [
        downsample(64, 4, batchnorm=False),
        downsample(128, 4),
        downsample(256, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
    ]
    
    up_stack = [
        upsample(512, 4, dropout=True),
        upsample(512, 4, dropout=True),
        upsample(512, 4),
        upsample(512, 4),
        upsample(256, 4),
        upsample(128, 4),
        upsample(64, 4),
    ]
    
    init = tf.random_normal_initializer(0., 0.02)
    last = Conv2DTranspose(3, 4, strides=2, padding="same", kernel_initializer=init, activation="tanh")
    
    x = inputs
    skips = []
    
    for down in down_stack:
        x = down(x)
        skips.append(x)
    
    skips = reversed(skips[:-1])
    
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = Concatenate()([x, skip])
    
    x = last(x)
    return Model(inputs=inputs, outputs=x)

# Discriminator model
def discriminator():
    init = tf.random_normal_initializer(0., 0.02)
    
    inp = Input(shape=[IMAGE_SIZE, IMAGE_SIZE, 3], name="input_image")
    tar = Input(shape=[IMAGE_SIZE, IMAGE_SIZE, 3], name="target_image")
    
    x = Concatenate()([inp, tar])
    
    down1 = downsample(64, 4, batchnorm=False)(x)
    down2 = downsample(128, 4)(down1)
    down3 = downsample(256, 4)(down2)
    
    zero_pad1 = ZeroPadding2D()(down3)
    conv = Conv2D(256, 4, strides=1, kernel_initializer=init, use_bias=False)(zero_pad1)
    leaky_relu = LeakyReLU()(conv)
    zero_pad2 = ZeroPadding2D()(leaky_relu)
    last = Conv2D(1, 4, strides=1, kernel_initializer=init)(zero_pad2)
    
    return Model(inputs=[inp, tar], outputs=last)

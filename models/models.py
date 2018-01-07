"""
file: models.py
author: Stephan Rasp (raspstephan@gmail.com)

Model definitions for GAN experiments.
We are using the Keras functional API.
"""

# Imports
import keras
import keras.backend as K
from keras.layers import *
from keras.optimizers import *
from keras.models import *


# Generator network
def upsample_block(x, filters, bn=False, kernel_size=5, activation='relu'):
    """

    """
    x = UpSampling2D(size=(2, 2))(x)
    return Convolution2D(filters, kernel_size=kernel_size, padding='same',
                         activation=activation)(x)


def create_generator(latent_size=100, first_conv_size=7, activation='relu',
                     bn=False, final_activation='tanh', kernel_size=5):
    """DCGAN
    To Do:
    - initialization
    """
    if activation == 'LeakyReLU': activation = LeakyReLU()

    # Random latent vector
    inp = Input(shape=(latent_size,))

    # One dense layer, then reshape to channels
    x = Dense(128 * (first_conv_size ** 2), activation=activation)(inp)
    if bn: x = BatchNormalization()(x)
    x = Reshape((first_conv_size, first_conv_size, 128))(x)

    # Double the image size twice
    x = upsample_block(x, 128, bn=bn, activation=activation,
                       kernel_size=kernel_size)
    x = upsample_block(x, 64, bn=bn, activation=activation,
                       kernel_size=kernel_size)

    # Reduce to one channel for the final image
    outp = Convolution2D(1, 2, padding='same', activation=final_activation)(x)

    return Model(inputs=inp, outputs=outp)


# Discriminator/critic network
def conv_block(x, filters, type='regular', activation='relu', kernel_size=5,
               dr=0, strides=2):
    if type == 'regular':
        x = Convolution2D(filters, kernel_size=kernel_size, strides=strides,
                          activation=activation, padding='same')(x)
        if not dr == 0: x = Dropout(dr)(x)
    elif type == 'max_pool':
        x = Convolution2D(filters, kernel_size=kernel_size, strides=1,
                          activation=activation, padding='same')(x)
        x = MaxPooling2D(pool_size=2)(x)
        if not dr == 0: x = Dropout(dr)(x)
    else:
        raise Exception('Wrong type for convolution block.')
    return x


def create_discriminator(filters=[128, 64], strides=[2, 2], dr=0,
                         conv_type='regular', activation='relu',
                         image_size=28, kernel_size=5,
                         final_activation='sigmoid'):
    """

    :return:
    """
    if activation == 'LeakyReLU': activation = LeakyReLU()

    # Input with channels last (Tensorflow convention)
    inp = Input(shape=(image_size, image_size, 1))

    # Convolution blocks, decrease image size twice
    x = inp
    for f, s in zip(filters, strides):
        x = conv_block(x, f, type=conv_type, activation=activation,
                       kernel_size=kernel_size, dr=dr, strides=s)

    # Final flattening and dense
    x = Flatten()(x)
    outp = Dense(1, activation=final_activation)(x)

    return Model(inputs=inp, outputs=outp)


# Create combined model
def compile_and_create_combined(G, D, latent_size=100):
    opt = Adam(lr=0.0002, beta_1=0.5)  # From original DCGAN paper
    G.compile(optimizer=opt, loss='mse')   # Loss here does not matter I guess
    D.compile(optimizer=opt, loss='binary_crossentropy')

    D.trainable = False
    inp_latent = Input(shape=(latent_size,))
    C = Model(inputs=inp_latent, outputs=D(G(inp_latent)))
    C.compile(optimizer=opt, loss='binary_crossentropy')
    return C






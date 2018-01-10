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


# Wasserstein loss function
def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)


# GENERATOR NETWORK
# Auxiliary functions
def upsample_block(x, filters, act_func, bn=False, kernel_size=5,
                   init='glorot_uniform', dr=0):
    """

    """
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(filters, kernel_size=kernel_size, padding='same',
               kernel_initializer=init)(x)
    x = act_func(x)
    if bn: x = BatchNormalization()(x)
    if not dr == 0: x = Dropout(dr)(x)
    return x


# Main function
def create_generator(filters=(128, 64), latent_size=100, first_conv_size=7,
                     activation='relu', bn=False, final_activation='tanh',
                     kernel_size=5, final_bias=True, init='glorot_uniform',
                     dr=0, final_kernel_size=1):
    """DCGAN
    """
    if activation == 'LeakyReLU': 
        act_func = LeakyReLU()
    else:
        act_func = Activation(activation)

    # Random latent vector
    inp = Input(shape=(latent_size,))

    # One dense layer, then reshape to channels
    x = Dense(filters[0] * (first_conv_size ** 2), 
              kernel_initializer=init)(inp)
    x = act_func(x)
    if bn: x = BatchNormalization()(x)
    x = Reshape((first_conv_size, first_conv_size, filters[0]))(x)

    # Double the image size twice
    for f in filters:
        x = upsample_block(x, f, act_func, bn=bn,
                           kernel_size=kernel_size, init=init, dr=dr)

    # Reduce to one channel for the final image
    outp = Conv2D(1, final_kernel_size, padding='same', 
                  activation=final_activation, use_bias=final_bias, 
                  kernel_initializer=init)(x)

    return Model(inputs=inp, outputs=outp)


# DISCRIMINATOR NETWORK
# Auxiliary functions
def conv_block(x, filters, act_func, type='regular', kernel_size=5,
               dr=0, strides=2, bn=False, init='glorot_uniform'):
    
    if type == 'regular':
        x = Conv2D(filters, kernel_size=kernel_size, strides=strides,
                   padding='same', kernel_initializer=init)(x)
        x = act_func(x)
        if bn: x = BatchNormalization()(x)
        if not dr == 0: x = Dropout(dr)(x)
    
    elif type == 'max_pool':
        x = Conv2D(filters, kernel_size=kernel_size, strides=1,
                          activation=activation, padding='same')(x)
        x = MaxPooling2D(pool_size=2)(x)
        if not dr == 0: x = Dropout(dr)(x)
    else:
        raise Exception('Wrong type for convolution block.')
    return x


# Main function
def create_discriminator(filters=(128, 64), strides=(2, 2), dr=0,
                         conv_type='regular', activation='relu',
                         image_size=28, kernel_size=5,
                         final_activation='sigmoid', bn=False,
                         init='glorot_uniform'):
    """
    No bn ever in first convolution, from [arXiv/1511.06434]
    :return:
    """
    if activation == 'LeakyReLU': 
        act_func = LeakyReLU()
    else:
        act_func = Activation(activation)

    # Input with channels last (Tensorflow convention)
    inp = Input(shape=(image_size, image_size, 1))

    # Convolution blocks, decrease image size twice
    x = inp
    for i, (f, s) in enumerate(zip(filters, strides)):
        x = conv_block(x, f, act_func, type=conv_type,
                       kernel_size=kernel_size, dr=dr, strides=s,
                       bn=bn if i > 0 else False, init=init)

    # Final flattening and dense
    x = Flatten()(x)
    outp = Dense(1, activation=final_activation, kernel_initializer=init)(x)

    return Model(inputs=inp, outputs=outp)







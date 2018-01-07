import keras
import tensorflow as tf
import matplotlib.pyplot as plt

def limit_mem():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # Allocates as much memory as needed.
    keras.backend.tensorflow_backend.set_session(tf.Session(config=config))

def plot_losses(dl, gl):
    plt.plot(dl, label='discriminator loss')
    plt.plot(gl, label='generator loss')
    plt.legend()
    plt.show()
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from skimage.measure import block_reduce
import seaborn as sns
sns.set_style('white')


def limit_mem():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # Allocates as much memory as needed.
    keras.backend.tensorflow_backend.set_session(tf.Session(config=config))


def plot_losses(train_history):
    plt.plot(train_history['train_discriminator_loss'],
             label='train discriminator loss')
    plt.plot(train_history['train_generator_loss'],
             label='train generator loss')
    plt.plot(train_history['test_discriminator_loss'],
             label='test discriminator loss')
    plt.plot(train_history['test_generator_loss'],
             label='test generator loss')
    plt.legend()
    plt.show()


def prep_mnist(X):
    X = (X.astype('float32') - 127.5) / 127.5
    return np.expand_dims(X, -1)


def halve_imsize(a):
    return block_reduce(a, (1, 2, 2), np.mean)


def log_normalize(x, c=1e-2):
    return (np.log(c + x) - np.log(c)) / 10.


def un_log_normalize(x, c=1e-2):
    return 10 * (np.exp(x + np.log(c)) - c)


def get_data(dataset, validation_split=0.2, normalize=False,
             remove_no_rain=True, halve_radar=False):

    if dataset == 'mnist':
        (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
        # Normalize the data
        X_train, X_test = (prep_mnist(X_train), prep_mnist(X_test))
        return X_train, X_test

    if dataset == 'radar':
        radar_data = np.load('/local/S.Rasp/tmp/radar_prepped.npy')

        if halve_radar:
            radar_data = halve_imsize(radar_data)

        # Add channel dimension
        radar_data = np.expand_dims(radar_data, -1)

        # Remove pictures without rain (or less than 0.1)
        if remove_no_rain:
            cov_frac = (np.sum(radar_data > 0.1, axis=(1, 2, 3)) /
                       (radar_data.shape[1]**2))
            radar_data = radar_data[cov_frac > 0]

        # Normalize if requested
        if normalize:
            radar_data = log_normalize(radar_data)

        # Split into test and validation set
        n = radar_data.shape[0]
        print('Number of radar stamps:', n)
        idxs = np.random.permutation(np.arange(n))
        train_idxs = idxs[int(n * validation_split):]
        test_idxs = idxs[:int(n * validation_split)]
        X_train = radar_data[train_idxs]
        X_test = radar_data[test_idxs]
        return X_train, X_test


def plot_stamps(data, normalize=False):
    levels = np.array([0, 0.1, 0.3, 1.0, 3.0, 10., 30., 100])
    if normalize:
        levels = log_normalize(levels)
    colors = [(1, 1, 1)] + sns.color_palette('cool', 6)
    cmap = matplotlib.colors.ListedColormap(colors)
    norm = matplotlib.colors.BoundaryNorm(levels, cmap.N)
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    axes_flat = list(np.ravel(axes))
    for i, ax in enumerate(axes_flat):
        ax.imshow(data[i], cmap=cmap, norm=norm)
    plt.show()


def create_noise(bs, latent_size, noise_shape='uniform'):
    if noise_shape == 'normal':
        return np.random.normal(0, 1, (bs, latent_size))
    elif noise_shape == 'uniform':
        return np.random.uniform(-1, 1, (bs, latent_size))
    else:
        raise ValueError('Wrong distribution for noise creation.')



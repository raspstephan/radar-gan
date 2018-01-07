"""
Training of the networks
"""
from keras.datasets import mnist
import numpy as np
from tqdm import tqdm
from PIL import Image


def prep_mnist(X):
    X = (X.astype('float32') - 127.5) / 127.5
    return np.expand_dims(X, -1)


def train(D, G, C, epochs, bs, batch_creation='complete_random',
          latent_size=100, image_dir='./images/', exp_name='test'):

    # Get the data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # Normalize the data
    X_train, X_test = (prep_mnist(X_train), prep_mnist(X_test))

    n_samples = X_train.shape[0]
    n_batches = n_samples // bs

    for e in tqdm(range(epochs)):
        dl, gl = [], []

        # Get random indices
        if batch_creation == 'complete_random':
            rand_idxs = np.arange(n_batches * bs)
            rand_idxs = np.random.permutation(rand_idxs)

        for b in range(n_batches):

            # TRAIN DISCRIMINATOR
            D.trainable = True

            # Get the real images
            if batch_creation == 'complete_random':
                real = X_train[rand_idxs[b*bs:(b+1)*bs]]
            elif batch_creation == 'random':
                real = X_train[np.random.randint(0, n_samples, bs)]
            elif batch_creation == 'complete':
                real = X_train[b*bs:(b+1)*bs]
            else:
                raise Exception('Wrong batch_creation type.')

            # Get some fake images
            latent_noise = np.random.uniform(-1, 1, (bs, latent_size))
            fake = G.predict_on_batch(latent_noise)

            # Concatenate inputs and targets
            # The real images get [1]s as labels, fake [0]s
            X_concat = np.concatenate([real, fake])
            y_concat = np.array([1] * bs + [0] * bs)

            # Train the discriminator
            dl.append(D.train_on_batch(X_concat, y_concat))


            # TRAIN GENERATOR
            D.trainable = False

            # Get noise input for twice the batch_size
            latent_noise = np.random.uniform(-1, 1, (2 * bs, latent_size))

            # Train the Generator
            gl.append(C.train_on_batch(latent_noise, np.array([1] * (2*bs))))


        # TODO Test loss

        # Create some sample images every epoch
        latent_noise = np.random.uniform(-1, 1, (9, latent_size))
        generated_images = np.squeeze(G.predict_on_batch(latent_noise))

        # From https://github.com/lukedeo/keras-acgan/blob/master/mnist_acgan.py
        img = (np.concatenate(
            [generated_images[i*3:(i+1)*3].reshape(-1, 28) for i in range(3)],
            axis=-1
        ) * 127.5 + 127.5).astype(np.uint8)
        Image.fromarray(img).save(image_dir + '/' + exp_name + '_' +
            'plot_epoch_{0:03d}_generated.png'.format(e))

    # Save weights

    return dl, gl






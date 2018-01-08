"""
Training of the networks
"""
from tqdm import tqdm
from PIL import Image
from collections import defaultdict
from .utils import *
import pdb


def train(D, G, C, epochs, bs, batch_creation='complete_random',
          latent_size=100, image_dir='./images/', exp_name='test',
          dataset='mnist', normalize_radar=False, remove_no_rain=True):
    """Train function, returns training history containing the losses.
    """
    # Some initial setup
    train_history = defaultdict(list)

    (X_train, y_train), (X_test, y_test) = get_data(
        dataset, normalize=normalize_radar, remove_no_rain=remove_no_rain
    )

    n_samples = X_train.shape[0]
    n_test = X_test.shape[0]
    image_size = X_train.shape[1]
    n_batches = n_samples // bs

    for e in tqdm(range(epochs)):
        dl, gl = [], []

        # Get random indices
        if batch_creation == 'complete_random':
            rand_idxs = np.arange(n_batches * bs)
            rand_idxs = np.random.permutation(rand_idxs)

        for b in range(n_batches):

            # STEP 1: TRAIN DISCRIMINATOR
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

            # STEP 2: TRAIN GENERATOR
            D.trainable = False

            # Get noise input for twice the batch_size
            latent_noise = np.random.uniform(-1, 1, (2 * bs, latent_size))

            # Train the Generator
            gl.append(C.train_on_batch(latent_noise, np.array([1] * (2*bs))))

        # COMPUTE LOSSES AFTER EACH EPOCH
        # First up the mean losses during training
        train_history['train_discriminator_loss'].append(np.mean(dl))
        train_history['train_generator_loss'].append(np.mean(gl))

        # Then get the scores for the test set
        latent_noise = np.random.uniform(-1, 1, (n_test, latent_size))
        fake = G.predict(latent_noise, batch_size=bs)
        X_concat = np.concatenate([X_test, fake])
        y_concat = np.array([1] * n_test + [0] * n_test)
        train_history['test_discriminator_loss'].append(
            D.evaluate(X_concat, y_concat, batch_size=bs, verbose=0)
        )
        latent_noise = np.random.uniform(-1, 1, (2 * n_test, latent_size))
        train_history['test_generator_loss'].append(
            C.evaluate(latent_noise, [1] * (2*n_test), batch_size=bs,
                       verbose=0)
        )

        # SAVE SOME IMAGES
        str = (image_dir + '/' + exp_name + '_' +
               'plot_epoch_{0:03d}_generated'.format(e))
        if dataset == 'mnist':
            # From https://github.com/lukedeo/keras-acgan/blob/master/mnist_acgan.py
            img = (np.concatenate(
                [fake[i*3:(i+1)*3, :, :, 0].reshape(-1, image_size)
                 for i in range(3)],
                axis=-1
            ) * 127.5 + 127.5).astype(np.uint8)
            Image.fromarray(img).save(str + '.png')
        if dataset == 'radar':
            np.save(str + '.npy', fake[:9])

    # TODO Save weights

    return train_history






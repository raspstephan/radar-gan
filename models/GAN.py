"""GAN class definition

"""

# Imports
from .models import *
from .utils import *
from collections import defaultdict
from tqdm import tqdm
from PIL import Image
import os
import pickle


# GAN class
class GAN(object):
    """GAN class"""
    def __init__(self, latent_size=100, image_size=None, exp_id='test',
                 img_dir='./images/', model_dir='./saved_models/',
                 wasserstein=False):
        """Initialize the variables"""
        # Initialize empty networks
        self.G = None
        self.D = None
        self.GD = None

        # Initialize the data variables
        self.X_train = None
        self.X_test = None
        self.n_train = None
        self.n_test = None

        # Some basic properties of the network
        self.dataset = None
        self.latent_size = latent_size
        self.image_size = image_size
        self.exp_id = exp_id
        self.img_dir = img_dir
        if not os.path.exists(img_dir): os.makedirs(img_dir)
        self.model_dir = model_dir
        if not os.path.exists(model_dir): os.makedirs(model_dir)
        self.wasserstein = wasserstein

        # Training information
        self.epoch_counter = 0
        self.train_history = defaultdict(list)

    def create_generator(self, **kwargs):
        """Create the generator network"""
        self.G = create_generator(latent_size=self.latent_size, **kwargs)

    def create_discriminator(self, **kwargs):
        """Create the discriminator network"""
        assert self.image_size is not None, 'Define image_size first!'
        self.D = create_discriminator(image_size=self.image_size, **kwargs)

    def compile(self):
        """Compile the networks and create the combined network"""
        if self.wasserstein:
            opt = RMSprop(lr=0.00005)
            loss = wasserstein_loss
        else:   # for Wasserstein distance
            opt = Adam(lr=0.0002, beta_1=0.5)  # From original DCGAN paper
            loss = 'binary_crossentropy'
        # Compile the individual models
        self.G.compile(optimizer=opt, loss='mse')  # Loss  does not matter
        self.D.compile(optimizer=opt, loss=loss)

        # Create and compile the combined model
        self.D.trainable = False
        inp_latent = Input(shape=(self.latent_size,))
        self.GD = Model(inputs=inp_latent, outputs=self.D(self.G(inp_latent)))
        self.GD.compile(optimizer=opt, loss=loss)

    def load_data(self, dataset, **kwargs):
        """Load the requested dataset"""
        self.dataset = dataset
        self.X_train, self.X_test = get_data(dataset, **kwargs)
        self.n_train, self.n_test = self.X_train.shape[0], self.X_test.shape[0]
        print('Train samples:', self.n_train, 'Test samples:', self.n_test)
        if self.image_size is None:
            self.image_size = self.X_train.shape[1]
        else:
            assert self.image_size == self.X_train.shape[1], \
                'image_size mismatch.'

    def train(self, epochs, bs, train_D_separately=False,
              noise_shape='uniform', n_disc=1):
        """Training operation"""
        n_batches = self.n_train // bs
        for e in tqdm(range(epochs)):
            dl, gl = [], []
            for b in range(n_batches):
                self.train_step(bs, dl, gl, train_D_separately, noise_shape,
                                n_disc)
            self.epoch_counter += 1

            # END OF EPOCH. COMPUTE AVERAGE AND TEST LOSSES
            self.train_history['train_discriminator_loss'].append(np.mean(dl))
            self.train_history['train_generator_loss'].append(np.mean(gl))

            fake = self.G.predict(
                create_noise(self.n_test, self.latent_size, noise_shape),
                batch_size=bs
            )
            X_concat = np.concatenate([self.X_test, fake])
            y_concat = np.array([1] * self.n_test + [0] * self.n_test)
            self.train_history['test_discriminator_loss'].append(
                self.D.evaluate(X_concat, y_concat, batch_size=bs, verbose=0)
            )
            self.train_history['test_generator_loss'].append(
                self.GD.evaluate(
                    create_noise(2*self.n_test, self.latent_size, noise_shape),
                    [1] * (2 * self.n_test),
                    batch_size=bs, verbose=0
                ))

            # Save images
            self.save_images(fake)

    def train_step(self, bs, dl, gl, train_D_separately, noise_shape,
                   n_disc):
        """One training step. May contain several discriminator steps."""

        # STEP 1: TRAIN DISCRIMINATOR
        self.D.trainable = True

        for i_disc in range(n_disc):
            # Get images
            real = self.X_train[np.random.randint(0, self.n_train, bs)]

            # Create fake images
            fake = self.G.predict_on_batch(
                create_noise(bs, self.latent_size, noise_shape)
            )

            # Concatenate real and fake images and train the discriminator
            if train_D_separately:
                # Train on real data first
                tmp = self.D.train_on_batch(real, np.array([1] * bs))
                # Then on fake data
                tmp += self.D.train_on_batch(fake, np.array([0] * bs))
                dl.append(tmp / 2.)
            else:
                X_concat = np.concatenate([real, fake])
                y_concat = np.array([1] * bs + [0] * bs)
                dl.append(self.D.train_on_batch(X_concat, y_concat))

        # STEP 2: TRAIN GENERATOR
        self.D.trainable = False
        gl.append(self.GD.train_on_batch(
            create_noise(2 * bs, self.latent_size, noise_shape),
            np.array([1] * (2 * bs))
        ))
        return dl, gl

    def save_images(self, fake):
        """Saves some fake images"""
        s = (self.img_dir + '/' + self.exp_id + '_' +
               'plot_epoch_{0:03d}_generated'.format(self.epoch_counter))
        if self.dataset == 'mnist':
            # From https://github.com/lukedeo/keras-acgan/blob/master/mnist_acgan.py
            img = (np.concatenate(
                [fake[i * 3:(i + 1) * 3, :, :, 0].reshape(-1, self.image_size)
                 for i in range(3)],
                axis=-1
            ) * 127.5 + 127.5).astype(np.uint8)
            Image.fromarray(img).save(s + '.png')
        if self.dataset == 'radar':
            np.save(s + '.npy', fake[:9])

    def save_models(self):
        """Saves models and training history"""
        s = self.model_dir + self.exp_id + '_'
        self.G.save(s + 'G.h5')
        self.D.save(s + 'D.h5')
        self.GD.save(s + 'GD.h5')
        # Save training history
        with open(s + 'history.pkl') as f:
            pickle.dump(self.train_history, f)
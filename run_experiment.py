"""Script to run GAN experiment.
"""
from models.GAN import GAN
from models.utils import *
from configargparse import ArgParser
import os
from keras.models import load_model

def main(inargs):
    os.environ["CUDA_VISIBLE_DEVICES"]=str(inargs.GPU)
    limit_mem()
    
    # Create GAN object
    gan = GAN(
        exp_id=inargs.exp_id, 
        wasserstein=inargs.wasserstein, 
        verbose=inargs.verbose
    )
    
    # Load dataset
    gan.load_data(
        inargs.dataset,
        normalize=inargs.normalize,
        halve_radar=inargs.halve_radar,
        data_dir=inargs.data_dir,
    )
    
    # Define networks
    if inargs.load_model:
        s = gan.model_dir + gan.exp_id + '_'
        gan.G = load_model(s + 'G.h5')
        gan.D = load_model(s + 'D.h5')
        train_fn = './saved_models/%s_history.pkl' % gan.exp_id
        with open(train_fn, 'rb') as f:
            gan.train_history = pickle.load(f)
        gan.epoch_counter = len(gan.train_history['train_discriminator_loss'])
                                
    else: 
        gan.create_generator(
            filters=inargs.G_filters, 
            first_conv_size=inargs.G_first_conv_size, 
            activation=inargs.G_activation,
            bn=inargs.G_bn,
            dr=inargs.G_dr,
            final_activation=inargs.G_final_activation,
            final_kernel_size=inargs.G_final_kernel_size,
        )
        gan.create_discriminator(
            filters=inargs.D_filters,
            strides=inargs.D_strides,
            activation=inargs.D_activation,
            bn=inargs.D_bn,
            dr=inargs.D_dr,
        )
    gan.G.summary()
    gan.D.summary()
    
    # Compile
    gan.compile()
    
    # Train
    gan.train(
        inargs.epochs,
        inargs.bs,
        inargs.train_D_separately,
        inargs.noise_shape,
    )
    
    # Save model
    gan.save_models()


if __name__ == '__main__':
    
    p = ArgParser()

    # Config file
    p.add(
        '-c', '--config',
        is_config_file=True,
        help='Config file path.'
    )

    # Directories and experiment name
    p.add_argument(
        '--data_dir',
        type=str,
        default='/home/ri27jiz/data/',
        help='Directory containing radar data. '
             'Default: /home/ri27jiz/data/',
    )
    p.add_argument(
        '--exp_id',
        type=str,
        default='test',
        help='Identifier of experiment. '
             'Default: test',
    )
    
    
    # General settings
    p.add_argument(
        '--verbose',
        type=int,
        default=0,
        help='Verbosity level. '
             'Default: 0',
    )
    p.add_argument(
        '--wasserstein',
        dest='wasserstein',
        action='store_true',
        help='If given, use wasserstein GAN.',
    )
    p.set_defaults(wasserstein=False)
    p.add_argument(
        '--GPU',
        type=int,
        default=0,
        help='Which GPU. '
             'Default: 0',
    )
    p.add_argument(
        '--load_model',
        dest='load_model',
        action='store_true',
        help='If given, load model.',
    )
    p.set_defaults(load_model=False)
    
    # Dataset settings
    p.add_argument(
        '--dataset',
        type=str,
        default='mnist',
        help='Dataset: mnist or radar. '
             'Default: mnist',
    )
    p.add_argument(
        '--normalize',
        type=int,
        default=1,
        help='Normalize radar data?',
    )
    p.add_argument(
        '--halve_radar',
        type=int,
        default=1,
        help='Halve radar data?',
    )
    
    # Training settings
    p.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of epochs to train. '
             'Default: 10',
    )
    p.add_argument(
        '--bs',
        type=int,
        default=128,
        help='Batch size. '
             'Default: 128',
    )
    p.add_argument(
        '--train_D_separately',
        type=int,
        default=1,
        help='If 1, train real and fake samples separately.',
    )
    p.add_argument(
        '--noise_shape',
        type=str,
        default='uniform',
        help='Latent noise shape. Default: uniform',
    )
    
    # Generator settings
    p.add_argument(
        '--G_filters',
        type=int,
        nargs='+',
        default=[128, 64],
        help='Filters in each convolution layer. '
             'Default: [128, 64]',
    )
    p.add_argument(
        '--G_first_conv_size',
        type=int,
        default=7,
        help='Size of first rescaled layer. '
             'Default: 7',
    )
    p.add_argument(
        '--G_activation',
        type=str,
        default='LeakyReLU',
        help='Activation function. '
             'Default: LeakyReLU',
    )
    p.add_argument(
        '--G_bn',
        type=int,
        default=1,
        help='Use batch norm?',
    )
    p.add_argument(
        '--G_dr',
        type=float,
        default=0,
        help='Dropout ratio, default: 0',
    )
    p.add_argument(
        '--G_final_activation',
        type=str,
        default='tanh',
        help='Final activation function. '
             'Default: tanh',
    )
    p.add_argument(
        '--G_final_kernel_size',
        type=int,
        default=1,
        help='Final kernel size. '
             'Default: 1',
    )
    
    # Discriminator settings
    p.add_argument(
        '--D_filters',
        type=int,
        nargs='+',
        default=[64, 128],
        help='Filters in each convolution layer. '
             'Default: [64, 128]',
    )
    p.add_argument(
        '--D_strides',
        type=int,
        nargs='+',
        default=[2, 2],
        help='Strides in each convolution layer. '
             'Default: [2, 2]',
    )
    p.add_argument(
        '--D_activation',
        type=str,
        default='LeakyReLU',
        help='Activation function. '
             'Default: LeakyReLU',
    )
    p.add_argument(
        '--D_bn',
        type=int,
        default=1,
        help='Use batch norm?',
    )
    p.add_argument(
        '--D_dr',
        type=float,
        default=0,
        help='Dropout ratio, default: 0',
    )

    args = p.parse_args()
    main(args)
def config(config_name='cifar_simple'):
    if config_name == 'cifar_simple':
        margs = {'num_block': 3,
                 'num_layer_per_block': 5,
                 'num_filter': [32, 64, 128],
                 'fc_dim': [512, 128],
                 'latent_dim': 32}
    elif config_name == 'cifar_complex':
        margs = {'num_block': 3,
                 'num_layer_per_block': 5,
                 'num_filter': [64, 128, 256],
                 'fc_dim': [1024, 256],
                 'latent_dim': 64}
    elif config_name == 'cifar_complex2':
        margs = {'num_block': 3,
                 'num_layer_per_block': 5,
                 'num_filter': [128, 256, 512],
                 'fc_dim': [1024, 256],
                 'latent_dim': 128}
    else:
        margs = {}
        print('No config named {0}.'.format(config_name))
    return margs


def data_folder(data_set):
    if data_set == 'Face':
        return 'D:/v-bindai/Projects/General_CVAE/data/Facescrub/crop'
    elif data_set == 'Bird':
        return 'D:/v-bindai/Projects/General_CVAE/data/cub200/imgs'
    elif data_set == 'Flower':
        return 'D:/v-bindai/Projects/General_CVAE/data/Flowers/imgs'
    else:
        return ''
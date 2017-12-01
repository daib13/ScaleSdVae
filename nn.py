import tensorflow as tf
from tensorflow.contrib import layers


def conv(name, x, num_filter, filter_size=[3, 3], stride=[1, 1, 1, 1], padding='SAME', reg=None):
    """ Convolution layer """
    filter_shape = [filter_size[0], filter_size[1],
                    int(x.get_shape()[-1]), num_filter]
    with tf.variable_scope(name):
        w = tf.get_variable('w', filter_shape, tf.float32,
                            layers.xavier_initializer_conv2d(), reg, True)
        b = tf.get_variable('b', [num_filter], tf.float32,
                            tf.zeros_initializer(), trainable=True)
        print(w.name)
        return tf.nn.conv2d(x, w, stride, padding) + b


def conv_norm(name, x, num_filter, filter_size=[3, 3], stride=[1, 1, 1, 1], padding='SAME', reg=None, activation_fn=None):
    """ Convolution with batch normalization """
    conv_x = conv(name, x, num_filter, filter_size, stride, padding, reg)
    return layers.batch_norm(conv_x, activation_fn=activation_fn)


def res_layer(name, x, filter_size=[3, 3], padding='SAME', reg=None, activation_fn=tf.nn.elu):
    """ Convolutional residule layer """
    num_filter = int(x.get_shape()[-1])
    stride = [1, 1, 1, 1]
    return x + conv_norm(name, x, num_filter, filter_size, stride, padding, reg, activation_fn)


def res_block(name, x, num_layer=2, num_filter=64, filter_size=[3, 3], padding='SAME', reg=None, activation_fn=tf.nn.elu):
    """ Residule block consisted of a convolution layer, which transforms a C1-channel feature map to a C2-channel feature map, and some residule layers (# is specified by num_layer) """
    stride = [1, 1, 1, 1]
    with tf.variable_scope(name):
        c = conv('init', x, num_filter, [1, 1], stride, padding, reg)
        for i in range(num_layer):
            c = res_layer('layer'+str(i), c, filter_size, padding, reg, activation_fn)
    return c

add a line
import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np


conv_w_init = layers.variance_scaling_initializer(3.0, uniform=True)
conv_b_init = tf.constant_initializer(0.2)


def inception(name, x, phase, k1_channel, k1_pool_channel=None, k3_channel1=None, k3_channel2=None, k5_channel1=None, k5_channel2=None, shortcut=False, reg=None, has_batch_norm=True):
    input_channel = int(x.get_shape()[-1])
    with tf.variable_scope(name + '_w'):
        total_channel = k1_channel
        with tf.variable_scope('k1_w'):
            k1_w = tf.get_variable('w', [1, 1, input_channel, k1_channel], tf.float32, conv_w_init, reg)
            k1_b = tf.get_variable('b', [k1_channel], tf.float32, conv_b_init)
        if k1_pool_channel is not None:
            with tf.variable_scope('k1_pool_w'):
                k1_pool_w = tf.get_variable('w', [1, 1, input_channel, k1_pool_channel], tf.float32, conv_w_init, reg)
                k1_pool_b = tf.get_variable('b', [k1_pool_channel], tf.float32, conv_b_init)
                total_channel += k1_pool_channel
        if k3_channel1 is not None and k3_channel2 is not None:
            with tf.variable_scope('k3_w'):
                k3_w1 = tf.get_variable('w1', [1, 1, input_channel, k3_channel1], tf.float32, conv_w_init, reg)
                k3_b1 = tf.get_variable('b1', [k3_channel1], tf.float32, conv_b_init)
                k3_w2 = tf.get_variable('w2', [3, 3, k3_channel1, k3_channel2], tf.float32, conv_w_init, reg)
                k3_b2 = tf.get_variable('b2', [k3_channel2], tf.float32, conv_b_init)
                total_channel += k3_channel2
        if k5_channel1 is not None and k5_channel2 is not None:
            with tf.variable_scope('k5_w'):
                k5_w1 = tf.get_variable('w1', [1, 1, input_channel, k5_channel1], tf.float32, conv_w_init, reg)
                k5_b1 = tf.get_variable('b1', [k5_channel1], tf.float32, conv_b_init)
                k5_w2 = tf.get_variable('w2', [5, 5, k5_channel1, k5_channel2], tf.float32, conv_w_init, reg)
                k5_b2 = tf.get_variable('b2', [k5_channel2], tf.float32, conv_b_init)
                total_channel += k5_channel2
        if shortcut == True:
            with tf.variable_scope('shortcut_w'):
                shortcut_w = tf.get_variable('w', [1, 1, input_channel, total_channel], tf.float32, conv_w_init, reg)
    with tf.name_scope(name):
        tensors = []
        with tf.name_scope('k1'):
            k1 = tf.nn.bias_add(tf.nn.conv2d(x, k1_w, [1, 1, 1, 1], 'SAME'), k1_b, name='k1') 
            if has_batch_norm:
                k1 = layers.batch_norm(k1, scale=True, is_training=phase)
            tensors.append(k1)

        if k1_pool_channel is not None:
            with tf.name_scope('k1_pool'):
                pool = tf.nn.max_pool(x, [1, 3, 3, 1], [1, 1, 1, 1], 'SAME', name='pool')
                k1_pool = tf.nn.bias_add(tf.nn.conv2d(pool, k1_pool_w, [1, 1, 1, 1], 'SAME'), k1_pool_b, name='k1_pool')
                if has_batch_norm:
                    k1_pool = layers.batch_norm(k1_pool, scale=True, is_training=phase)
                tensors.append(k1_pool)

        if k3_channel1 is not None and k3_channel2 is not None:
            with tf.name_scope('k3'):
                k3_1 = tf.nn.bias_add(tf.nn.conv2d(x, k3_w1, [1, 1, 1, 1], 'SAME'), k3_b1, name='k3_1')
                k3_1_relu = tf.nn.relu(k3_1, 'k3_1_relu')
                k3_2 = tf.nn.bias_add(tf.nn.conv2d(k3_1_relu, k3_w2, [1, 1, 1, 1], 'SAME'), k3_b2, name='k3_2')
                if has_batch_norm:
                    k3_2 = layers.batch_norm(k3_2, scale=True, is_training=phase)
                tensors.append(k3_2)

        if k5_channel1 is not None and k5_channel2 is not None:
            with tf.name_scope('k5'):
                k5_1 = tf.nn.bias_add(tf.nn.conv2d(x, k5_w1, [1, 1, 1, 1], 'SAME'), k5_b1, name='k5_1')
                k5_1_relu = tf.nn.relu(k5_1, 'k5_1_relu')
                k5_2 = tf.nn.bias_add(tf.nn.conv2d(k5_1_relu, k5_w2, [1, 1, 1, 1], 'SAME'), k5_b2, name='k5_2')
                if has_batch_norm:
                    k5_2 = layers.batch_norm(k5_2, scale=True, is_training=phase)
                tensors.append(k5_2)

        if shortcut == True:
            with tf.name_scope('shortcut'):
                x_transform = tf.nn.conv2d(x, shortcut_w, [1, 1, 1, 1], 'SAME', name='shortcut')
            if has_batch_norm:
                x_transform = layers.batch_norm(x_transform, scale=True, is_training=phase)
                
        with tf.name_scope('output'):
            if len(tensors) > 1:
                concat = tf.concat(tensors, -1, 'concat')
            else:
                concat = k1
            output = tf.nn.relu(concat, 'output')
            if shortcut == True:
                output = tf.add(output, x_transform, 'res_output')
    return output


def upsample_map(x, scale=2, method=tf.image.ResizeMethod.BILINEAR):
    shape = x.get_shape().as_list()
    assert len(shape) == 4
    new_shape = [scale * shape[1], scale * shape[2]]
    return tf.image.resize_images(x, new_shape, method)


def deconv_block(name, x, phase, output_channel, kernel_size, upsample=False, shortcut=False, reg=None, activation_fn=None):
    input_shape = x.get_shape()
    input_channel = input_shape[-1]
    with tf.variable_scope(name + '_w'):
        w = tf.get_variable('w', [kernel_size, kernel_size, input_channel, output_channel], tf.float32, conv_w_init, reg)
        b = tf.get_variable('b', [output_channel], tf.float32, conv_b_init, reg)
        if shortcut == True and activation_fn is not None:
            w_shortcut = tf.get_variable('w_shortcut', [1, 1, input_channel, output_channel], tf.float32, conv_w_init, reg)
    with tf.name_scope(name):
        output = tf.nn.bias_add(tf.nn.conv2d(x, w, [1, 1, 1, 1], 'SAME'), b, name='conv') 
        output = layers.batch_norm(output, scale=True, is_training=phase)
        if activation_fn is not None:
            output = activation_fn(output)
        if shortcut == True and activation_fn is not None:
            x_transform = tf.nn.conv2d(x, w_shortcut, [1, 1, 1, 1], 'SAME')
            x_transform = layers.batch_norm(x_transform, scale=True, is_training=phase)
            output = tf.add(output, x_transform, 'conv_res')
        if upsample == True:
            output = upsample_map(output)
    return output
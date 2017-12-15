import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np


def inception(name, x, k1_channel, k1_pool_channel, k3_channel1, k3_channel2, k5_channel1, k5_channel2, reg=None):
    input_channel = int(x.get_shape()[-1]
    with tf.variable_scope(name + '_w'):
        with tf.variable_scope('k1_w'):
            k1_w = tf.get_variable('w', [1, 1, input_channel, k1_channel], tf.float32, layers.xavier_initializer_conv2d(), reg)
            k1_b = tf.get-variable('b', [k1_channel], tf.float32, tf.zeros_initializer())
        with tf.variable_scope('k1_pool_w'):
            k1_pool_w = tf.get_variable('w', [1, 1, input_channel, k1_pool_channel], tf.float32, layers.xavier_initializer_conv2d(), reg)
            k1_pool_b = tf.get_variable('b', [k1_pool_channel], tf.float32, tf.zeros_initializer())
        with tf.variable_scope('k3_w'):
            k3_w1 = tf.get_variable('w1', [1, 1, input_channel, k3_channel1], tf.float32, layers.xavier_initializer_conv2d(), reg)
            k3_b1 = tf.get_variable('b1', [k3_channel1], tf.float32, tf.zeros_initializer())
            k3_w2 = tf.get_variable('w2', [3, 3, k3_channel1, k3_channel2], tf.float32, layers.xavier_initializer_conv2d(), reg)
            k3_b2 = tf.get_variable('b2', [k3_channel2], tf.float32, tf.zeros_initializer())
        with tf.variable_scope('k5_w'):
            k5_w1 = tf.get_variable('w1', [1, 1, input_channel, k5_channel1], tf.float32, layers.xavier_initializer_conv2d(), reg)
            k5_b1 = tf.get_variable('b1', [k5_channel1], tf.float32, tf.zeros_initializer())
            k5_w2 = tf.get_variable('w2', [5, 5, k5_channel1, k5_channel2], tf.float32, layers.xavier_initializer_conv2d(), reg)
            k5_b2 = tf.get_variable('b2', [k5_channel2], tf.float32, tf.zeros_initializer())
    with tf.name_scope(name):
        with tf.name_scope('k1'):
            k1 = tf.nn.bias_add(tf.nn.conv2d(x, k1_w, [1, 1, 1, 1], 'SAME'), k1_b, name='k1') 
        with tf.name_scope('k1_pool'):
            pool = tf.nn.max_pool(x)

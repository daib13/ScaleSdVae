import tensorflow as tf
from tensorflow.contrib import layers
from inception import inception, upsample_map, conv_w_init, conv_b_init, deconv_block


def decoder(name, z, phase, shortcut=False, reg=None, layer_per_scale=2):
    latent_dim = int(z.get_shape()[-1])
    with tf.name_scope(name):
        # fc
        with tf.variable_scope(name + '_w'):
            with tf.variable_scope('de_fc_w'):
                fc1_w = tf.get_variable('w1', [latent_dim, 512], tf.float32, conv_w_init, reg)
                fc1_b = tf.get_variable('b1', [512], tf.float32, tf.zeros_initializer())            
                fc2_w = tf.get_variable('w2', [512, 4096], tf.float32, conv_w_init, reg)
                fc2_b = tf.get_variable('b2', [4096], tf.float32, tf.zeros_initializer())
                if shortcut:
                    fc1_shortcut_w = tf.get_variable('sw1', [latent_dim, 512], tf.float32, conv_w_init, reg)
                    fc2_shortcut_w = tf.get_variable('sw2', [512, 4096], tf.float32, conv_w_init, reg)
        with tf.name_scope('fc'):
            fc1 = tf.nn.bias_add(tf.matmul(z, fc1_w), fc1_b, name='fc1')
            fc1_relu = tf.nn.relu(fc1, name='fc1_relu')
            if shortcut:
                fc1_shortcut = tf.matmul(z, fc1_shortcut_w, name='fc1_shortcut')
                fc1_relu = tf.add(fc1_relu, fc1_shortcut, 'fc1_add')
            fc2 = tf.nn.bias_add(tf.matmul(fc1_relu, fc2_w), fc2_b, name='fc2')
            fc2_relu = tf.nn.relu(fc2, name='fc2_relu')
            if shortcut:
                fc2_shortcut = tf.matmul(fc1_relu, fc2_shortcut_w, name='fc2_shortcut')
                fc2_relu = tf.add(fc2_relu, fc2_shortcut, 'fc2_add')

        # feature2
        feature = tf.reshape(fc2_relu, [-1, 2, 2, 1024], 'feature2')

        # feature4
        with tf.name_scope('feature4'):
            feature = upsample_map(feature)
            with tf.variable_scope(name + '_w'):
                feature = inception('feature4', feature, phase, 1024, shortcut=shortcut, reg=reg)

        # feature8
        with tf.name_scope('feature8'):
            feature = upsample_map(feature)
            with tf.variable_scope(name + '_w'):
                for i in range(layer_per_scale):
                    feature = inception('feature8_' + str(i), feature, phase, 256, 128, 160, 320, 32, 128, shortcut, reg)

        # feature16
        with tf.name_scope('feature16'):
            feature = upsample_map(feature)
            with tf.variable_scope(name + '_w'):
                for i in range(layer_per_scale):
                    feature = inception('feature16_' + str(i), feature, phase, 128, 64, 128, 224, 32, 96, shortcut, reg)

        # feature32
        with tf.name_scope('feature32'):
            feature = upsample_map(feature)
            with tf.variable_scope(name + '_w'):
                for i in range(layer_per_scale):
                    feature = inception('feature32_' + str(i), feature, phase, 64, 32, 64, 112, 16, 48, shortcut, reg)

        # feature64
        with tf.name_scope('feature64'):
            feature = upsample_map(feature)
            with tf.variable_scope(name + '_w'):
                for i in range(layer_per_scale):
                    feature = inception('feature64_' + str(i), feature, phase, 32, 16, 32, 56, 8, 24, shortcut, reg)

        # feature128
        with tf.name_scope('feature128'):
            feature = upsample_map(feature)
            with tf.variable_scope(name + '_w'):
                for i in range(layer_per_scale):
                    feature = inception('feature128_' + str(i), feature, phase, 16, 8, 16, 28, 4, 12, shortcut, reg)

        with tf.variable_scope(name + '_w'):
            with tf.variable_scope('x_hat_w'):
                x_hat_w = tf.get_variable('w', [1, 1, 64, 3], tf.float32, conv_w_init, reg)
                x_hat_b = tf.get_variable('b', [3], tf.float32, tf.zeros_initializer())
        with tf.name_scope('x_hat'):
            x_hat_logit = tf.nn.bias_add(tf.nn.conv2d(feature, x_hat_w, [1, 1, 1, 1], 'SAME'), x_hat_b, name='x_hat_logit')
            x_hat = tf.nn.sigmoid(x_hat_logit)
            
    return x_hat


def decoder_simple1(name, z, phase, shortcut=False, reg=None):
    latent_dim = int(z.get_shape()[-1])
    with tf.name_scope(name):
        # fc
        with tf.variable_scope('de_fc_w'):
            fc1_w = tf.get_variable('w1', [latent_dim, 512], tf.float32, conv_w_init, reg)
            fc1_b = tf.get_variable('b1', [512], tf.float32, tf.zeros_initializer())            
            fc2_w = tf.get_variable('w2', [512, 4096], tf.float32, conv_w_init, reg)
            fc2_b = tf.get_variable('b2', [4096], tf.float32, tf.zeros_initializer())
            if shortcut:
                fc1_shortcut_w = tf.get_variable('sw1', [latent_dim, 512], tf.float32, conv_w_init, reg)
                fc2_shortcut_w = tf.get_variable('sw2', [512, 4096], tf.float32, conv_w_init, reg)
        with tf.name_scope('fc'):
            fc1 = tf.nn.bias_add(tf.matmul(z, fc1_w), fc1_b, name='fc1')
            fc1_relu = tf.nn.relu(fc1, name='fc1_relu')
            if shortcut:
                fc1_shortcut = tf.matmul(z, fc1_shortcut_w, name='fc1_shortcut')
                fc1_relu = tf.add(fc1_relu, fc1_shortcut, 'fc1_add')
            fc2 = tf.nn.bias_add(tf.matmul(fc1_relu, fc2_w), fc2_b, name='fc2')
            fc2_relu = tf.nn.relu(fc2, name='fc2_relu')
            if shortcut:
                fc2_shortcut = tf.matmul(fc1_relu, fc2_shortcut_w, name='fc2_shortcut')
                fc2_relu = tf.add(fc2_relu, fc2_shortcut, 'fc2_add')

         # feature4
        feature = tf.reshape(fc2_relu, [-1, 4, 4, 256], 'feature2')

        # feature8
        feature = deconv_block('feature8', feature, phase, 256, 3, True, shortcut, reg, tf.nn.relu)

        # feature16
        feature = deconv_block('feature16', feature, phase, 256, 3, True, shortcut, reg, tf.nn.relu)

        # feature32
        feature = deconv_block('feature32', feature, phase, 128, 5, True, shortcut, reg, tf.nn.relu)

        # feature64
        feature = deconv_block('feature64', feature, phase, 92, 5, True, shortcut, reg, tf.nn.relu)

        # feature128
        feature = deconv_block('feature128_1', feature, phase, 64, 5, True, shortcut, reg, tf.nn.relu)
        feature = deconv_block('feature128_2', feature, phase, 3, 5, False, shortcut, reg, tf.nn.sigmoid)
    return feature
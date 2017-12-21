import tensorflow as tf
from tensorflow.contrib import layers
from inception import inception, conv_w_init, conv_b_init


def encoder_googlenet(name, x, phase, latent_dim, reg=None, log_scale=None):
    input_channel = x.get_shape()[-1]

    with tf.name_scope(name):
        # block 1      
        with tf.variable_scope('block1_w'):
            conv1_w = tf.get_variable('w', [7, 7, input_channel, 64], tf.float32, conv_w_init, reg)
            conv1_b = tf.get_variable('b', [64], tf.float32, conv_b_init)
        with tf.name_scope('block1'):
            conv1 = tf.nn.bias_add(tf.nn.conv2d(x, conv1_w, [1, 2, 2, 1], 'SAME'), conv1_b, name='conv1')
            conv1 = layers.batch_norm(conv1, scale=True, is_training=phase)
            conv1_relu = tf.nn.relu(conv1, name='conv1_relu')
            pool1 = tf.nn.max_pool(conv1_relu, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME', name='pool1')
            lrn1 = tf.nn.local_response_normalization(pool1, alpha=0.0001, beta=0.75, name='lrn1')

        # block 2
        with tf.variable_scope('block2_w'):
            conv2_1_w = tf.get_variable('w1', [1, 1, 64, 64], tf.float32, conv_w_init, reg)
            conv2_1_b = tf.get_variable('b1', [64], tf.float32, conv_w_init)
            conv2_2_w = tf.get_variable('w2', [3, 3, 64, 192], tf.float32, conv_w_init, reg)
            conv2_2_b = tf.get_variable('b2', [192], tf.float32, conv_b_init)
        with tf.name_scope('block2'):
            conv2_1 = tf.nn.bias_add(tf.nn.conv2d(lrn1, conv2_1_w, [1, 1, 1, 1], 'SAME'), conv2_1_b, name='conv2_1')
            conv2_1 = layers.batch_norm(conv2_1, scale=True, is_training=phase)
            conv2_1_relu = tf.nn.relu(conv2_1, name='conv2_1_relu')
            conv2_2 = tf.nn.bias_add(tf.nn.conv2d(conv2_1_relu, conv2_2_w, [1, 1, 1, 1], 'SAME'), conv2_2_b, name='conv2_2')
            conv2_2 = layers.batch_norm(conv2_2, scale=True, is_training=phase)
            conv2_2_relu = tf.nn.relu(conv2_2, name='conv2_2_relu')
            lrn2 = tf.nn.local_response_normalization(conv2_2_relu, alpha=0.0001, beta=0.75, name='lrn2')
            pool2 = tf.nn.max_pool(lrn2, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME', name='pool2')
        
        # block 3 (inception 3)
        with tf.name_scope('block3'):
            inception3_a = inception('inception3_a', pool2, phase, 64, 32, 96, 128, 16, 32, reg) #256
            inception3_b = inception('inception3_b', inception3_a, phase, 128, 64, 128, 224, 32, 96, reg) #512
            pool3 = tf.nn.max_pool(inception3_b, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME', name='pool3') # 8*8

        # block 4 (inception 4)
        with tf.name_scope('block4'):
            inception4_a = inception('inception4_a', pool3, phase, 192, 64, 96, 208, 16, 48, reg)
            inception4_b = inception('inception4_b', inception4_a, phase, 160, 64, 112, 224, 24, 64, reg)
            inception4_c = inception('inception4_c', inception4_b, phase, 128, 64, 128, 256, 24, 64, reg)
            inception4_d = inception('inception4_d', inception4_c, phase, 112, 64, 144, 288, 32, 64, reg)
            inception4_e = inception('inception4_e', inception4_d, phase, 256, 128, 160, 512, 32, 128, reg)
            pool4 = tf.nn.max_pool(inception4_e, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME', name='pool4')

        # block 5 (inception 5)
        with tf.name_scope('block5'):
            inception5_a = inception('inception5_a', pool4, phase, 256, 128, 160, 320, 32, 128, reg)
            inception5_b = inception('inception5_b', inception5_a, phase, 384, 128, 192, 384, 48, 128, reg)
            pool5 = tf.nn.avg_pool(inception5_b, [1, 4, 4, 1], [1, 4, 4, 1], 'SAME', name='pool5')
            pool5_flatten = tf.reshape(pool5, [tf.shape(pool5, out_type=tf.int32)[0], -1], 'pool5_flatten')

        # fc
        with tf.variable_scope('en_fc_w'):
            fc6_w = tf.get_variable('w1', [1024, 512], tf.float32, conv_w_init, reg)
            fc6_b = tf.get_variable('b1', [512], tf.float32, tf.zeros_initializer())
            fc7_w = tf.get_variable('w2', [512, 256], tf.float32, conv_w_init, reg)
            fc7_b = tf.get_variable('b2', [256], tf.float32, tf.zeros_initializer())
        with tf.name_scope('fc'):
            fc6 = tf.nn.bias_add(tf.matmul(pool5_flatten, fc6_w), fc6_b, name='fc6')
            fc6_relu = tf.nn.relu(fc6, 'fc6_relu')
            fc7 = tf.nn.bias_add(tf.matmul(fc6_relu, fc7_w), fc7_b, name='fc7')
            fc7_relu = tf.nn.relu(fc7, 'fc7_relu')

        # latent
        with tf.variable_scope('latent_w'):
            mu_w = tf.get_variable('mu_w', [256, latent_dim], tf.float32, conv_w_init, reg)
            mu_b = tf.get_variable('mu_b', [latent_dim], tf.float32, tf.zeros_initializer())
            logsd_w = tf.get_variable('logsd_w', [256, latent_dim], tf.float32, layers.variance_scaling_initializer(3.0/400.0, uniform=True), reg)
            logsd_b = tf.get_variable('logsd_b', [latent_dim], tf.float32, tf.constant_initializer(0.0))
        with tf.name_scope('latent'):
            mu = tf.nn.bias_add(tf.matmul(fc7_relu, mu_w), mu_b, name='mu')
            scale_logsd = tf.nn.bias_add(tf.matmul(fc7_relu, logsd_w), logsd_b, name='scale_logsd')
            if log_scale is None:
                sd = tf.exp(scale_logsd)
            else:
                sd = tf.exp(scale_logsd + log_scale / 2.0)
    
    return mu, scale_logsd, sd
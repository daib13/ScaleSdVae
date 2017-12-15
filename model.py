from nn import res_block, dense, upsample
import tensorflow as tf
from tensorflow.contrib import layers


class ConvVae:
    def __init__(self, num_block, num_layer_per_block, num_filter, filter_size=[3, 3], padding='SAME',
                 fc_dim=[1024], latent_dim=64, weight_decay=0.0, activation_fn=tf.nn.elu):
        self.img_shape = [None, 32, 32, 3]
        assert len(num_filter) == num_block
        self.num_block = num_block
        self.num_layer_per_block = num_layer_per_block
        self.num_filter = num_filter
        self.filter_size = filter_size
        self.padding = padding
        self.fc_dim = fc_dim
        self.num_fc_layer = len(fc_dim)
        self.latent_dim = latent_dim
        self.reg = layers.l2_regularizer(weight_decay)
        self.activation_fn = activation_fn
        self.smallest_size = int(32 / pow(2, num_block))

        self.__build_net()

    def __build_net(self):
        self.__build_encoder()
        self.__build_decoder()
        self.__build_loss()
        self.__build_summary()
        self.__build_optimizer()

    def __build_encoder(self):
        with tf.name_scope('x'):
            self.x = tf.placeholder(tf.float32, self.img_shape)
            self.batch_size = tf.shape(self.x, out_type=tf.int32)[0]
        feature_map = self.x
        with tf.name_scope('encoder'):
            with tf.variable_scope('encoder_w'):
                for i_block in range(self.num_block):
                    with tf.name_scope('block' + str(i_block)):
                        feature_map = res_block('block' + str(i_block) + '_w', feature_map, self.num_layer_per_block,
                                                self.num_filter[i_block], self.filter_size, self.padding, self.reg, self.activation_fn)
                    with tf.name_scope('pool' + str(i_block)):
                        feature_map = tf.nn.max_pool(
                            feature_map, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
                for i_fc in range(self.num_fc_layer):
                    with tf.name_scope('fc' + str(i_fc)):
                        feature_map = dense('fc' + str(i_fc) + '_w', feature_map,
                                            self.fc_dim[i_fc], self.reg, self.activation_fn)
                with tf.name_scope('mu_z'):
                    self.mu_z = dense('mu_z_w', feature_map,
                                      self.latent_dim, self.reg)
                with tf.name_scope('sd_z'):
                    self.logsd_z = dense(
                        'logsd_z_w', feature_map, self.latent_dim, self.reg)
                    self.sd_z = tf.exp(self.logsd_z)

    def __build_decoder(self):
        with tf.name_scope('sample'):
            self.noise = tf.random_normal(
                [self.batch_size, self.latent_dim], 0.0, 1.0, tf.float32)
            self.z = self.noise * self.sd_z + self.mu_z
        feature_map = self.z
        with tf.name_scope('decoder'):
            with tf.variable_scope('decoder_w'):
                for i_fc in range(self.num_fc_layer):
                    with tf.name_scope('fc' + str(i_fc)):
                        feature_map = dense('fc' + str(i_fc) + '_w', feature_map,
                                            self.fc_dim[-1 - i_fc], self.reg, self.activation_fn)
                feature_map_dim = self.num_filter[-1] * \
                    self.smallest_size * self.smallest_size
                with tf.name_scope('fc' + str(self.num_fc_layer)):
                    feature_map = dense('fc' + str(self.num_fc_layer) + '_w', feature_map,
                                        feature_map_dim, self.reg, self.activation_fn)
                    feature_map = tf.reshape(
                        feature_map, [-1, self.smallest_size, self.smallest_size, self.num_filter[-1]])
                for i_block in range(self.num_block):
                    with tf.name_scope('upsample' + str(i_block)):
                        feature_map = upsample(feature_map)
                    with tf.name_scope('block' + str(i_block)):
                        if i_block == self.num_block - 1:
                            num_filter = 3
                        else:
                            num_filter = self.num_filter[-2 - i_block]
                        feature_map = res_block('block' + str(i_block) + '_w', feature_map, self.num_layer_per_block,
                                                num_filter, self.filter_size, self.padding, self.reg, self.activation_fn)
                with tf.name_scope('x_hat'):
                    self.x_hat = tf.nn.sigmoid(feature_map)
                with tf.variable_scope('log_gamma'):
                    self.log_gamma = tf.get_variable('log_gamma', [], tf.float32, tf.zeros_initializer(), trainable=True)
                with tf.name_scope('gamma'):
                    self.gamma = tf.exp(self.log_gamma)

    def __build_loss(self):
        self.kl_loss = tf.reduce_sum(tf.square(self.mu_z) + tf.square(self.sd_z) - 2 * self.logsd_z - 1) / 2.0 / tf.cast(self.batch_size, tf.float32)
        self.gen_loss = tf.reduce_sum(tf.square(self.x_hat - self.x) / 2.0 / self.gamma + self.log_gamma / 2.0) / tf.cast(self.batch_size, tf.float32)
        self.loss = self.kl_loss + self.gen_loss

    def __build_summary(self):
        tf.summary.scalar('kl_loss', self.kl_loss)
        tf.summary.scalar('gen_loss', self.gen_loss)
        tf.summary.scalar('loss', self.loss)
        tf.summary.image('recon', self.x_hat)
        tf.summary.scalar('gamma', self.gamma)
        tf.summary.histogram('sd_z', self.sd_z)
        self.summary = tf.summary.merge_all()

    def __build_optimizer(self):
        with tf.name_scope('optimizer'):
            with tf.variable_scope('optimizer_moments'):
                self.lr = tf.placeholder(tf.float32, [], 'lr')
                self.global_step = tf.get_variable('global_step', [], tf.int32, tf.zeros_initializer(), trainable=False)
                self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss, self.global_step)


class ConvAe:
    def __init__(self, num_block, num_layer_per_block, num_filter, filter_size=[3, 3], padding='SAME',
                 fc_dim=[1024], latent_dim=64, weight_decay=0.0, activation_fn=tf.nn.elu):
        self.img_shape = [None, 32, 32, 3]
        assert len(num_filter) == num_block
        self.num_block = num_block
        self.num_layer_per_block = num_layer_per_block
        self.num_filter = num_filter
        self.filter_size = filter_size
        self.padding = padding
        self.fc_dim = fc_dim
        self.num_fc_layer = len(fc_dim)
        self.latent_dim = latent_dim
        self.reg = layers.l2_regularizer(weight_decay)
        self.activation_fn = activation_fn
        self.smallest_size = int(32 / pow(2, num_block))

        self.__build_net()

    def __build_net(self):
        self.__build_encoder()
        self.__build_decoder()
        self.__build_loss()
        self.__build_summary()
        self.__build_optimizer()

    def __build_encoder(self):
        with tf.name_scope('x'):
            self.x = tf.placeholder(tf.float32, self.img_shape)
            self.batch_size = tf.shape(self.x, out_type=tf.int32)[0]
        feature_map = self.x
        with tf.name_scope('encoder'):
            with tf.variable_scope('encoder_w'):
                for i_block in range(self.num_block):
                    with tf.name_scope('block' + str(i_block)):
                        feature_map = res_block('block' + str(i_block) + '_w', feature_map, self.num_layer_per_block,
                                                self.num_filter[i_block], self.filter_size, self.padding, self.reg, self.activation_fn)
                    with tf.name_scope('pool' + str(i_block)):
                        feature_map = tf.nn.max_pool(
                            feature_map, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
                for i_fc in range(self.num_fc_layer):
                    with tf.name_scope('fc' + str(i_fc)):
                        feature_map = dense('fc' + str(i_fc) + '_w', feature_map,
                                            self.fc_dim[i_fc], self.reg, self.activation_fn)
                with tf.name_scope('z'):
                    self.z = dense('z_w', feature_map, self.latent_dim, self.reg)

    def __build_decoder(self):
        feature_map = self.z
        with tf.name_scope('decoder'):
            with tf.variable_scope('decoder_w'):
                for i_fc in range(self.num_fc_layer):
                    with tf.name_scope('fc' + str(i_fc)):
                        feature_map = dense('fc' + str(i_fc) + '_w', feature_map,
                                            self.fc_dim[-1 - i_fc], self.reg, self.activation_fn)
                feature_map_dim = self.num_filter[-1] * \
                    self.smallest_size * self.smallest_size
                with tf.name_scope('fc' + str(self.num_fc_layer)):
                    feature_map = dense('fc' + str(self.num_fc_layer) + '_w', feature_map,
                                        feature_map_dim, self.reg, self.activation_fn)
                    feature_map = tf.reshape(
                        feature_map, [-1, self.smallest_size, self.smallest_size, self.num_filter[-1]])
                for i_block in range(self.num_block):
                    with tf.name_scope('upsample' + str(i_block)):
                        feature_map = upsample(feature_map)
                    with tf.name_scope('block' + str(i_block)):
                        if i_block == self.num_block - 1:
                            num_filter = 3
                        else:
                            num_filter = self.num_filter[-2 - i_block]
                        feature_map = res_block('block' + str(i_block) + '_w', feature_map, self.num_layer_per_block,
                                                num_filter, self.filter_size, self.padding, self.reg, self.activation_fn)
                with tf.name_scope('x_hat'):
                    self.x_hat = tf.nn.sigmoid(feature_map)
                with tf.variable_scope('log_gamma'):
                    self.log_gamma = tf.get_variable('log_gamma', [], tf.float32, tf.zeros_initializer(), trainable=True)
                with tf.name_scope('gamma'):
                    self.gamma = tf.exp(self.log_gamma)

    def __build_loss(self):
        self.loss = tf.reduce_sum(tf.square(self.x_hat - self.x) / 2.0 / self.gamma + self.log_gamma / 2.0) / tf.cast(self.batch_size, tf.float32)

    def __build_summary(self):
        tf.summary.scalar('loss', self.loss)
        tf.summary.image('recon', self.x_hat)
        tf.summary.scalar('gamma', self.gamma)
        self.summary = tf.summary.merge_all()

    def __build_optimizer(self):
        with tf.name_scope('optimizer'):
            with tf.variable_scope('optimizer_moments'):
                self.lr = tf.placeholder(tf.float32, [], 'lr')
                self.global_step = tf.get_variable('global_step', [], tf.int32, tf.zeros_initializer(), trainable=False)
                self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss, self.global_step)
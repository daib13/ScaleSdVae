import tensorflow as tf
from tensorflow.contrib import layers


def dense(name, x, output_dim, activation_fn=None, regularizer=None):
    input_shape = x.get_shape()
    input_dim = input_shape[-1]
    if not len(input_shape) == 2:
        x = tf.reshape(x, [-1, input_shape[-1]])
    with tf.variable_scope(name + '_w'):
        w = tf.get_variable('w', [input_dim, output_dim],
                            tf.float32, layers.xavier_initializer(), regularizer, True)
        b = tf.get_variable('b', [output_dim], tf.float32,
                            tf.zeros_initializer(), trainable=True)
    with tf.variable_scope(name):
        y = tf.matmul(x, w) + b
        if activation_fn is not None:
            y = activation_fn(y)
    return y


class VaeSimple:
    def __init__(self, input_dim, latent_dim, encoder_dim, decoder_dim, sample_num, activation_fn=tf.nn.tanh, regularizer=0.0, beta1=0.9, beta2=0.999, epsilon=1e-4):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder_dim = encoder_dim
        self.encoder_num = len(encoder_dim)
        self.decoder_dim = decoder_dim
        self.decoder_num = len(decoder_dim)
        self.sample_num = sample_num
        self.activation_fn = activation_fn
        self.regularizer = layers.l2_regularizer(regularizer)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.__build_encoder()
        self.__build_decoder()
        self.__build_loss()
        self.__build_summary()
        self.__build_optimizer()
        self.__build_nll()

    def __build_encoder(self):
        with tf.name_scope('x'):
            self.x = tf.placeholder(
                tf.float32, shape=[None, self.input_dim], name='x')
            self.batch_size = tf.cast(tf.shape(self.x)[0], tf.int32)
            x_ref = tf.random_uniform(
                [self.batch_size, self.input_dim, 1], 0.0, 1.0, dtype=tf.float32, name='x_ref')
            x_reshape = tf.reshape(
                self.x, [self.batch_size, self.input_dim, 1], name='x_reshape')
            x_concat = tf.concat([x_ref, x_reshape], 2)
            self.x_binary = tf.cast(tf.arg_max(
                x_concat, 2), tf.float32, 'x_binary')
        previous_tensor = self.x_binary
        with tf.name_scope('encoder'):
            with tf.variable_scope('encoder_w'):
                for i in range(self.encoder_num):
                    previous_tensor = dense(
                        'layer' + str(i), previous_tensor, self.encoder_dim[i], self.activation_fn, self.regularizer)
                self.mu_z = dense('mu_z', previous_tensor,
                                  self.latent_dim, regularizer=self.regularizer)
                self.logsd_z = dense(
                    'logsd_z', previous_tensor, self.latent_dim, regularizer=self.regularizer)
                with tf.name_scope('sd_z'):
                    self.sd_z = tf.exp(self.logsd_z)

    def __build_decoder(self):
        with tf.name_scope('sample'):
            duplicate_mu_z = tf.reshape(tf.tile(tf.reshape(self.mu_z, [self.batch_size, 1, self.latent_dim]), [
                                        1, self.sample_num, 1]), [self.batch_size * self.sample_num, self.latent_dim])
            duplicate_sd_z = tf.reshape(tf.tile(tf.reshape(self.sd_z, [self.batch_size, 1, self.latent_dim]), [
                                        1, self.sample_num, 1]), [self.batch_size * self.sample_num, self.latent_dim])
            self.noise = tf.random_normal(
                [self.batch_size * self.sample_num, self.latent_dim])
            self.z = self.noise * duplicate_sd_z + duplicate_mu_z
        previous_tensor = self.z
        with tf.name_scope('decoder'):
            with tf.variable_scope('decoder_w'):
                for i in range(self.decoder_num):
                    previous_tensor = dense(
                        'layer' + str(i), previous_tensor, self.decoder_dim[i], self.activation_fn, self.regularizer)
                self.x_hat_logit = dense(
                    'x_hat_logit', previous_tensor, self.input_dim, regularizer=self.regularizer)
                with tf.name_scope('x_hat'):
                    self.x_hat = tf.nn.sigmoid(self.x_hat_logit)

    def __build_loss(self):
        with tf.name_scope('loss'):
            self.kl_loss = tf.reduce_sum(tf.square(self.mu_z) + tf.square(
                self.sd_z) - 2.0 * self.logsd_z - 1.0) / tf.cast(self.batch_size, tf.float32) / 2.0
            duplicate_x = tf.reshape(tf.tile(tf.reshape(self.x_binary, [self.batch_size, 1, self.input_dim]), [
                                     1, self.sample_num, 1]), [self.batch_size * self.sample_num, self.input_dim])
            self.logit_x = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=duplicate_x, logits=self.x_hat_logit)
            self.gen_loss = tf.reduce_sum(
                self.logit_x) / tf.cast(self.batch_size, tf.float32) / self.sample_num
            self.loss = self.kl_loss + self.gen_loss

    def __build_summary(self):
        with tf.name_scope('summary'):
            tf.summary.scalar('kl_loss', self.kl_loss)
            tf.summary.scalar('gen_loss', self.gen_loss)
            tf.summary.scalar('loss', self.loss)
            self.summary = tf.summary.merge_all()

    def __build_optimizer(self):
        with tf.name_scope('optimizer'):
            with tf.variable_scope('optimizer_moments'):
                self.lr = tf.placeholder(tf.float32, [], 'lr')
                self.global_step = tf.get_variable(
                    'global_step', [], tf.int32, tf.zeros_initializer(), trainable=False)
                self.optimizer = tf.train.AdamOptimizer(
                    self.lr, self.beta1, self.beta2, self.epsilon).minimize(self.loss, self.global_step)

    def __build_nll(self):
        with tf.name_scope('nll'):
            duplicate_logsd_z = tf.reshape(tf.tile(tf.reshape(self.logsd_z, [self.batch_size, 1, self.latent_dim]), [
                                           1, self.sample_num, 1]), [self.batch_size * self.sample_num, self.latent_dim])
            self.logit_z = duplicate_logsd_z + \
                tf.square(self.noise) / 2 - tf.square(self.z) / 2
            self.logit = tf.reshape(tf.reduce_sum(
                self.logit_z, -1) + tf.reduce_sum(self.logit_x, -1), [self.batch_size, self.sample_num])
            self.logit_max = tf.reduce_max(self.logit, 1)
            self.logit_max_tile = tf.tile(tf.reshape(
                self.logit_max, [self.batch_size, 1]), [1, self.sample_num])
            self.res = tf.reduce_sum(
                tf.exp(self.logit - self.logit_max_tile), 1)
            self.nll = - self.logit_max - tf.log(self.res)

    def partial_train(self, x, lr, sess, writer):
        loss, _, summary = sess.run(
            [self.loss, self.optimizer, self.summary], feed_dict={self.x: x, self.lr: lr})
        writer.add_summary(summary, self.global_step.eval(sess))
        return loss

    def test_nll(self, x, sess):
        nll = sess.run([self.nll], feed_dict={self.x: x})
        return nll

    def save_to(self, saver, sess, save_path):
        saver.save(sess, save_path)

    def restore_from(self, saver, sess, restore_path):
        saver.restore(sess, restore_path)


class VaeScale(VaeSimple):
    def __init__(self, input_dim, latent_dim, encoder_dim, decoder_dim, sample_num, activation_fn=tf.nn.tanh, regularizer=0.0, beta1=0.9, beta2=0.999, epsilon=1e-4, gamma=1):
        self.gamma = gamma
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder_dim = encoder_dim
        self.encoder_num = len(encoder_dim)
        self.decoder_dim = decoder_dim
        self.decoder_num = len(decoder_dim)
        self.sample_num = sample_num
        self.activation_fn = activation_fn
        self.regularizer = layers.l2_regularizer(regularizer)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.__build_encoder()
        self.__build_decoder()
        self.__build_loss()
        self.__build_summary()
        self.__build_optimizer()
        self.__build_nll()

    def __build_encoder(self):
        with tf.name_scope('x'):
            self.x = tf.placeholder(
                tf.float32, shape=[None, self.input_dim], name='x')
            self.batch_size = tf.cast(tf.shape(self.x)[0], tf.int32)
            x_ref = tf.random_uniform(
                [self.batch_size, self.input_dim, 1], 0.0, 1.0, dtype=tf.float32, name='x_ref')
            x_reshape = tf.reshape(
                self.x, [self.batch_size, self.input_dim, 1], name='x_reshape')
            x_concat = tf.concat([x_ref, x_reshape], 2)
            self.x_binary = tf.cast(tf.arg_max(
                x_concat, 2), tf.float32, 'x_binary')
        previous_tensor = self.x_binary
        with tf.name_scope('encoder'):
            with tf.variable_scope('encoder_w'):
                for i in range(self.encoder_num):
                    previous_tensor = dense(
                        'layer' + str(i), previous_tensor, self.encoder_dim[i], self.activation_fn, self.regularizer)
                self.mu_z = dense('mu_z', previous_tensor,
                                  self.latent_dim, regularizer=self.regularizer)
                self.scale_logsd_z = dense(
                    'scale_logsd_z', previous_tensor, self.latent_dim, regularizer=self.regularizer)
                with tf.name_scope('sd_z'):
                    self.scale_sd_z = tf.exp(self.scale_logsd_z)
                    self.logsd_z = self.scale_logsd_z + \
                        0.5 * tf.log(self.gamma)
                    self.sd_z = self.scale_sd_z * tf.sqrt(self.gamma)

    def __build_decoder(self):
        with tf.name_scope('sample'):
            duplicate_mu_z = tf.reshape(tf.tile(tf.reshape(self.mu_z, [self.batch_size, 1, self.latent_dim]), [
                                        1, self.sample_num, 1]), [self.batch_size * self.sample_num, self.latent_dim])
            duplicate_sd_z = tf.reshape(tf.tile(tf.reshape(self.sd_z, [self.batch_size, 1, self.latent_dim]), [
                                        1, self.sample_num, 1]), [self.batch_size * self.sample_num, self.latent_dim])
            self.noise = tf.random_normal(
                [self.batch_size * self.sample_num, self.latent_dim])
            self.z = self.noise * duplicate_sd_z + duplicate_mu_z
        previous_tensor = self.z
        with tf.name_scope('decoder'):
            with tf.variable_scope('decoder_w'):
                for i in range(self.decoder_num):
                    previous_tensor = dense(
                        'layer' + str(i), previous_tensor, self.decoder_dim[i], self.activation_fn, self.regularizer)
                self.x_hat_logit = dense(
                    'x_hat_logit', previous_tensor, self.input_dim, regularizer=self.regularizer)
                with tf.name_scope('x_hat'):
                    self.x_hat = tf.nn.sigmoid(self.x_hat_logit)

    def __build_loss(self):
        with tf.name_scope('loss'):
            self.kl_loss = tf.reduce_sum(tf.square(self.mu_z) + tf.square(
                self.sd_z) - 2 * self.scale_logsd_z - tf.log(self.gamma) - 1.0) / tf.cast(self.batch_size, tf.float32) / 2.0
            duplicate_x = tf.reshape(tf.tile(tf.reshape(self.x_binary, [self.batch_size, 1, self.input_dim]), [
                                     1, self.sample_num, 1]), [self.batch_size * self.sample_num, self.input_dim])
            self.logit_x = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=duplicate_x, logits=self.x_hat_logit)
            self.gen_loss = tf.reduce_sum(
                self.logit_x) / tf.cast(self.batch_size, tf.float32) / self.sample_num
            self.loss = self.kl_loss + self.gen_loss

    def __build_summary(self):
        with tf.name_scope('summary'):
            tf.summary.scalar('kl_loss', self.kl_loss)
            tf.summary.scalar('gen_loss', self.gen_loss)
            tf.summary.scalar('loss', self.loss)
            self.summary = tf.summary.merge_all()

    def __build_optimizer(self):
        with tf.name_scope('optimizer'):
            with tf.variable_scope('optimizer_moments'):
                self.lr = tf.placeholder(tf.float32, [], 'lr')
                self.global_step = tf.get_variable(
                    'global_step', [], tf.int32, tf.zeros_initializer(), trainable=False)
                self.optimizer = tf.train.AdamOptimizer(
                    self.lr, self.beta1, self.beta2, self.epsilon).minimize(self.loss, self.global_step)

    def __build_nll(self):
        with tf.name_scope('nll'):
            duplicate_logsd_z = tf.reshape(tf.tile(tf.reshape(self.logsd_z, [self.batch_size, 1, self.latent_dim]), [
                                           1, self.sample_num, 1]), [self.batch_size * self.sample_num, self.latent_dim])
            self.logit_z = duplicate_logsd_z + \
                tf.square(self.noise) / 2 - tf.square(self.z) / 2
            self.logit = tf.reshape(tf.reduce_sum(
                self.logit_z, -1) + tf.reduce_sum(self.logit_x, -1), [self.batch_size, self.sample_num])
            self.logit_max = tf.reduce_max(self.logit, 1)
            self.logit_max_tile = tf.tile(tf.reshape(
                self.logit_max, [self.batch_size, 1]), [1, self.sample_num])
            self.res = tf.reduce_sum(
                tf.exp(self.logit - self.logit_max_tile), 1)
            self.nll = - self.logit_max - tf.log(self.res)
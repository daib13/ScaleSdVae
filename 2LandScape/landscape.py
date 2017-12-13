import tensorflow as tf
from tensorflow.contrib import layers
import os
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt


class Landscape:
    def __init__(self, x_dim, z_dim, kappa, encoder_dim, decoder_dim,
                 sample_num, batch_size=100, log_gamma=0.0, gamma_trainable=True):
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.kappa = kappa
        self.encoder_dim = encoder_dim
        self.encoder_num = len(encoder_dim)
        self.decoder_dim = decoder_dim
        self.decoder_num = len(decoder_dim)
        self.sample_num = sample_num
        self.batch_size = batch_size
        self.batch_size = batch_size
        self.init_log_gamma = log_gamma
        self.gamma_trainable = gamma_trainable

        self.__build_syn_net()
        self.__build_encoder()
        self.__build_decoder()

        self.__build_loss()
        self.__build_summary()
        self.__build_optimizer()

        self.__build_test_encoder()
        self.__build_test_decoder()
        self.__build_test_loss()

    def __build_syn_net(self):
        self.weights_syn = dict()
        previous_dim = self.z_dim
        with tf.variable_scope('syn_weight'):
            for i in range(self.decoder_num):
                with tf.variable_scope('layer' + str(i)):
                    self.weights_syn['layer' + str(i) + '_w'] = tf.get_variable('w', [previous_dim, self.decoder_dim[i]],
                                                                                tf.float32, tf.random_normal_initializer(0, (2.0 / previous_dim)**0.5), trainable=False)
                    self.weights_syn['layer' + str(i) + '_b'] = tf.get_variable(
                        'b', [self.decoder_dim[i]], tf.float32, tf.zeros_initializer(), trainable=False)
                    previous_dim = self.decoder_dim[i]
            with tf.variable_scope('x_hat'):
                self.weights_syn['x_hat_w'] = tf.get_variable(
                    'w', [previous_dim, self.x_dim], tf.float32, tf.random_normal_initializer(0, (2.0 / previous_dim)**0.5), trainable=False)
                self.weights_syn['x_hat_b'] = tf.get_variable(
                    'b', [self.x_dim], tf.float32, tf.zeros_initializer(), trainable=False)
        with tf.name_scope('syn_net'):
#            self.seed = tf.random_normal(
#                [self.batch_size, self.z_dim], 0.0, 1.0, tf.float32)
            self.seed = tf.placeholder(tf.float32, [self.batch_size, self.z_dim], 'seed')
            previous_tensor = self.seed
            for i in range(self.decoder_num):
                with tf.name_scope('layer' + str(i)):
                    previous_tensor = tf.matmul(
                        previous_tensor, self.weights_syn['layer' + str(i) + '_w']) + self.weights_syn['layer' + str(i) + '_b']
                    previous_tensor = tf.nn.relu(previous_tensor)
            with tf.name_scope('x'):
                self.x = tf.matmul(
                    previous_tensor, self.weights_syn['x_hat_w']) + self.weights_syn['x_hat_b']

    def __build_encoder(self):
        self.weights_enc = dict()
        previous_dim = self.x_dim
        with tf.variable_scope('encoder_weight'):
            for i in range(self.encoder_num):
                with tf.variable_scope('layer' + str(i)):
                    self.weights_enc['layer' + str(i) + '_w'] = tf.get_variable('w', [previous_dim, self.encoder_dim[i]],
                                                                                tf.float32, tf.random_normal_initializer(0, (2.0 / previous_dim)**0.5), trainable=True)
                    self.weights_enc['layer' + str(i) + '_b'] = tf.get_variable(
                        'b', [self.encoder_dim[i]], tf.float32, tf.zeros_initializer(), trainable=True)
                    previous_dim = self.encoder_dim[i]
            with tf.variable_scope('mu_z'):
                self.weights_enc['mu_z_w'] = tf.get_variable(
                    'w', [previous_dim, self.kappa], tf.float32, tf.random_normal_initializer(0, (2.0 / previous_dim)**0.5), trainable=True)
                self.weights_enc['mu_z_b'] = tf.get_variable(
                    'b', [self.kappa], tf.float32, tf.zeros_initializer(), trainable=True)
            with tf.variable_scope('logsd_z_w'):
                self.weights_enc['logsd_z_w'] = tf.get_variable(
                    'w', [previous_dim, self.kappa], tf.float32, tf.random_normal_initializer(0, (2.0 / previous_dim)**0.5), trainable=True)
                self.weights_enc['logsd_z_b'] = tf.get_variable(
                    'b', [self.kappa], tf.float32, tf.zeros_initializer(), trainable=True)
        previous_tensor = self.x
        with tf.name_scope('encoder'):
            for i in range(self.encoder_num):
                with tf.name_scope('layer' + str(i)):
                    previous_tensor = tf.matmul(previous_tensor, self.weights_enc['layer' + str(
                        i) + '_w']) + self.weights_enc['layer' + str(i) + '_b']
                    previous_tensor = tf.nn.relu(previous_tensor)
            self.h_e = previous_tensor
            with tf.name_scope('mu_z'):
                self.mu_z = tf.matmul(
                    previous_tensor, self.weights_enc['mu_z_w']) + self.weights_enc['mu_z_b']
            with tf.name_scope('sd_z'):
                self.logsd_z = tf.matmul(
                    previous_tensor, self.weights_enc['logsd_z_w']) + self.weights_enc['logsd_z_b']
                self.sd_z = tf.exp(self.logsd_z)
        self.w_mu = self.weights_enc['mu_z_w']
        self.b_mu = self.weights_enc['mu_z_b']
        self.w_sd = self.weights_enc['logsd_z_w']
        self.b_sd = self.weights_enc['logsd_z_b']

    def __build_decoder(self):
        self.weights_dec = dict()
        previous_dim = self.kappa
        with tf.variable_scope('decoder_weight'):
            for i in range(self.decoder_num):
                with tf.variable_scope('layer' + str(i)):
                    self.weights_dec['layer' + str(i) + '_w'] = tf.get_variable('w', [previous_dim, self.decoder_dim[i]],
                                                                                tf.float32, tf.random_normal_initializer(0, (2.0 / previous_dim)**0.5), trainable=True)
                    self.weights_dec['layer' + str(i) + '_b'] = tf.get_variable(
                        'b', [self.decoder_dim[i]], tf.float32, tf.zeros_initializer(), trainable=True)
                    previous_dim = self.decoder_dim[i]
            with tf.variable_scope('x_hat'):
                self.weights_dec['x_hat_w'] = tf.get_variable(
                    'w', [previous_dim, self.x_dim], tf.float32, tf.random_normal_initializer(0, (2.0 / previous_dim)**0.5), trainable=True)
                self.weights_dec['x_hat_b'] = tf.get_variable(
                    'b', [self.x_dim], tf.float32, tf.zeros_initializer(), trainable=True)
            with tf.variable_scope('log_gamma'):
                self.log_gamma = tf.get_variable('log_gamma', [], tf.float32, tf.constant_initializer(
                    self.init_log_gamma), trainable=self.gamma_trainable)
        with tf.name_scope('sample'):
            duplicate_mu_z = tf.reshape(tf.tile(tf.reshape(self.mu_z, [self.batch_size, 1, self.kappa]), [
                                        1, self.sample_num, 1]), [self.batch_size * self.sample_num, self.kappa])
            duplicate_sd_z = tf.reshape(tf.tile(tf.reshape(self.sd_z, [self.batch_size, 1, self.kappa]), [
                                        1, self.sample_num, 1]), [self.batch_size * self.sample_num, self.kappa])
            epsilon = tf.random_normal(
                [self.batch_size * self.sample_num, self.kappa], 0.0, 1.0, tf.float32)
            self.z = epsilon * duplicate_sd_z + duplicate_mu_z
        previous_tensor = self.z
        with tf.name_scope('decoder'):
            for i in range(self.decoder_num):
                with tf.name_scope('layer' + str(i)):
                    previous_tensor = tf.matmul(previous_tensor, self.weights_dec['layer' + str(
                        i) + '_w']) + self.weights_dec['layer' + str(i) + '_b']
                    if i == 0:
                        self.h_d = previous_tensor
                    previous_tensor = tf.nn.relu(previous_tensor)
            with tf.name_scope('x_hat'):
                self.x_hat = tf.matmul(
                    previous_tensor, self.weights_dec['x_hat_w']) + self.weights_dec['x_hat_b']
            with tf.name_scope('gamma'):
                self.gamma = tf.exp(self.log_gamma)
        if self.decoder_num == 0:
            self.w_o = self.weights_dec['x_hat_w']
        else:
            self.w_o = self.weights_dec['layer0_w']

    def __build_loss(self):
        with tf.name_scope('loss'):
            self.kl_loss = tf.reduce_sum(tf.square(
                self.mu_z) + tf.square(self.sd_z) - 2.0 * self.logsd_z - 1.0) / self.batch_size / 2.0
            self.duplicate_x = tf.reshape(tf.tile(tf.reshape(self.x, [self.batch_size, 1, self.x_dim]), [
                1, self.sample_num, 1]), [self.batch_size * self.sample_num, self.x_dim])
            self.gen_loss = tf.reduce_sum(tf.square(
                self.duplicate_x - self.x_hat) / self.gamma + self.log_gamma) / self.batch_size / self.sample_num / 2.0
            self.loss = self.kl_loss + self.gen_loss

    def __build_test_encoder(self):
        with tf.name_scope('test_encoder'):
            if self.encoder_num == 0:
                previous_dim = self.x_dim
            else:
                previous_dim = self.encoder_dim[self.encoder_num - 1]
            self.w_mu_test = tf.placeholder(
                tf.float32, [previous_dim, self.kappa], 'w_mu')
            self.b_mu_test = tf.placeholder(tf.float32, [self.kappa], 'b_mu')
            self.mu_z_test = tf.matmul(
                self.h_e, self.w_mu_test) + self.b_mu_test
            self.w_sd_test = tf.placeholder(
                tf.float32, [previous_dim, self.kappa], 'w_sd')
            self.b_sd_test = tf.placeholder(tf.float32, [self.kappa], 'b_sd')
            self.logsd_z_test = tf.matmul(
                self.h_e, self.w_sd_test) + self.b_sd_test
            self.sd_z_test = tf.exp(self.logsd_z_test)

    def __build_test_decoder(self):
        with tf.name_scope('sample_test'):
            duplicate_mu_z = tf.reshape(tf.tile(tf.reshape(self.mu_z_test, [self.batch_size, 1, self.kappa]), [
                                        1, self.sample_num, 1]), [self.batch_size * self.sample_num, self.kappa])
            duplicate_sd_z = tf.reshape(tf.tile(tf.reshape(self.sd_z_test, [self.batch_size, 1, self.kappa]), [
                                        1, self.sample_num, 1]), [self.batch_size * self.sample_num, self.kappa])
            epsilon = tf.random_normal(
                [self.batch_size * self.sample_num, self.kappa], 0.0, 1.0, tf.float32)
            self.z_test = epsilon * duplicate_sd_z + duplicate_mu_z
        previous_tensor = self.z_test
        with tf.name_scope('decoder_test'):
            for i in range(self.decoder_num):
                with tf.name_scope('layer' + str(i)):
                    if i == 0:
                        self.w_o_test = tf.placeholder(
                            tf.float32, [self.kappa, self.decoder_dim[i]], 'w_o')
                        previous_tensor = tf.matmul(
                            previous_tensor, self.w_o_test) + self.weights_dec['layer0_b']
                    else:
                        previous_tensor = tf.matmul(previous_tensor, self.weights_dec['layer' + str(
                            i) + '_w']) + self.weights_dec['layer' + str(i) + '_b']
                    previous_tensor = tf.nn.relu(previous_tensor)
            with tf.name_scope('x_hat'):
                if self.decoder_num == 0:
                    self.w_o_test = tf.placeholder(
                        tf.float32, [self.kappa, self.x_dim], 'w_o')
                    self.x_hat_test = tf.matmul(
                        previous_tensor, self.w_o_test) + self.weights_dec['x_hat_b']
                else:
                    self.x_hat_test = tf.matmul(
                        previous_tensor, self.weights_dec['x_hat_w']) + self.weights_dec['x_hat_b']

    def __build_test_loss(self):
        with tf.name_scope('loss_test'):
            kl_loss = tf.reduce_sum(tf.square(self.mu_z_test) + tf.square(
                self.sd_z_test) - 2.0 * self.logsd_z_test - 1.0) / self.batch_size / 2.0
            distance = tf.square(self.duplicate_x - self.x_hat_test)
            gamma = tf.reduce_mean(distance)
            gen_loss = tf.reduce_sum(
                distance / self.gamma + tf.log(gamma)) / self.batch_size / self.sample_num / 2.0
            self.loss_test = kl_loss + gen_loss

    def __build_summary(self):
        with tf.name_scope('summary'):
            tf.summary.scalar('kl_loss', self.kl_loss)
            tf.summary.scalar('gen_loss', self.gen_loss)
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('gamma', self.gamma)
            self.summary = tf.summary.merge_all()

    def __build_optimizer(self):
        with tf.name_scope('optimizer'):
            with tf.name_scope('optimizer_moments'):
                self.lr = tf.placeholder(tf.float32, [], 'lr')
                self.global_step = tf.get_variable(
                    'global_step', [], tf.int32, tf.zeros_initializer(), trainable=False)
                self.optimizer = tf.train.AdamOptimizer(
                    self.lr).minimize(self.loss, self.global_step)


def main():
    x_dim = 400
    z_dim = 20
    kappa = 30
    encoder_dim = [200, 200, 200]
    decoder_dim = [200, 200, 200]
    sample_num = 20
    iteration_num = 100000
    learning_rate = 0.001

    model = Landscape(x_dim, z_dim, kappa, encoder_dim,
                      decoder_dim, sample_num)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter('graph')
        if not os.path.exists('model'):
            os.mkdir('model')

        sess.run(tf.global_variables_initializer())
#        for i in range(iteration_num):
#            batch_loss, _, summary = sess.run(
#                [model.loss, model.optimizer, model.summary], feed_dict={model.lr: learning_rate})
#            writer.add_summary(summary, model.global_step.eval(sess))
#            if i % 100 == 99:
#                print('Iter = {0}, loss = {1}.'.format(i, batch_loss))

#        saver.save(sess, 'model/model.ckpt')
        saver.restore(sess, 'model/model.ckpt')

        w_mu = model.w_mu.eval(sess)
        b_mu = model.b_mu.eval(sess)
        w_sd = model.w_sd.eval(sess)
        b_sd = model.b_sd.eval(sess)
        w_o = model.w_o.eval(sess)

        scale = np.arange(-0.5, 1.51, 0.05)
        num_test = scale.shape[0]
        loss = np.zeros([kappa, num_test])
        seed = np.random.normal(0.0, 1.0, [100, z_dim])
        for i in range(kappa):
            for j in range(num_test):
                w_mu_test = deepcopy(w_mu)
                b_mu_test = deepcopy(b_mu)
                w_sd_test = deepcopy(w_sd)
                b_sd_test = deepcopy(b_sd)
                w_o_test = deepcopy(w_o)

                w_mu_test[:, i] *= scale[j]
                b_mu_test[i] *= scale[j]
                w_sd_test[:, i] *= scale[j]
                b_sd_test[i] *= scale[j]
                w_o_test[i, :] *= scale[j]

                loss[i, j] = sess.run(model.loss_test, feed_dict={model.w_mu_test: w_mu_test,
                                                                  model.b_mu_test: b_mu_test,
                                                                  model.w_sd_test: w_sd_test,
                                                                  model.b_sd_test: b_sd_test,
                                                                  model.w_o_test: w_o_test,
                                                                  model.seed: seed})
                
            plt.plot(scale, loss[i, :])
            plt.show()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    main()
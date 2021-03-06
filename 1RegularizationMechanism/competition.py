import tensorflow as tf
import os
import numpy as np
import pickle
import scipy.io
import sys


class VaeManifold:
    def __init__(self, x_dim, z_dim, kappa, encoder_dim, decoder_dim, sample_num, batch_size=100, log_gamma=0.0, gamma_trainable=True):
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.kappa = kappa
        self.encoder_dim = encoder_dim
        self.encoder_num = len(encoder_dim)
        self.decoder_dim = decoder_dim
        self.decoder_num = len(decoder_dim)
        self.sample_num = sample_num
        self.batch_size = batch_size
        self.init_log_gamma = log_gamma
        self.gamma_trainable = gamma_trainable

        self.__build_syn_net()
        self.__build_encoder()
        self.__build_decoder()

        self.__build_loss()
        self.__build_gradient()
        self.__build_summary()
        self.__build_optimizer()

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
            self.seed = tf.random_normal(
                [self.batch_size, self.z_dim], 0.0, 1.0, tf.float32)
            previous_tensor = self.seed
            for i in range(self.decoder_num):
                with tf.name_scope('layer' + str(i)):
                    previous_tensor = tf.matmul(
                        previous_tensor, self.weights_syn['layer' + str(i) + '_w']) + self.weights_syn['layer' + str(i) + '_b']
                    previous_tensor = tf.nn.relu(previous_tensor)
            with tf.name_scope('x_gen'):
                self.x_gen = tf.matmul(
                    previous_tensor, self.weights_syn['x_hat_w']) + self.weights_syn['x_hat_b']

    def __build_encoder(self):
        self.weights_enc = dict()
        previous_dim = self.x_dim
        self.x = tf.placeholder(tf.float32, [self.batch_size, self.x_dim], 'x')
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
                if self.decoder_num == 0:
                    self.h_d = self.x_hat
            with tf.name_scope('gamma'):
                self.gamma = tf.exp(self.log_gamma)

    def __build_loss(self):
        with tf.name_scope('loss'):
            self.kl_loss = tf.reduce_sum(tf.square(
                self.mu_z) + tf.square(self.sd_z) - 2.0 * self.logsd_z - 1.0) / self.batch_size / 2.0
            duplicate_x = tf.reshape(tf.tile(tf.reshape(self.x, [self.batch_size, 1, self.x_dim]), [
                                     1, self.sample_num, 1]), [self.batch_size * self.sample_num, self.x_dim])
            self.gen_loss = tf.reduce_sum(tf.square(
                duplicate_x - self.x_hat) / self.gamma + self.log_gamma) / self.batch_size / self.sample_num / 2.0
            self.loss = self.kl_loss + self.gen_loss

    def __build_gradient(self):
        with tf.name_scope('gradient'):
            self.gradient = dict()
            dh_d = tf.gradients(self.loss, self.h_d)
            self.gradient['h_d'] = tf.reduce_sum(tf.square(dh_d[0])) / self.batch_size / self.sample_num
            if self.decoder_num == 0:
                dw_o = tf.gradients(self.loss, self.weights_dec['x_hat_w'])
            else:
                dw_o = tf.gradients(self.loss, self.weights_dec['layer0_w'])
            self.gradient['w_o'] = tf.reshape(tf.reduce_sum(tf.square(dw_o[0]), 1), [1, -1])
            dz = tf.gradients(self.loss, self.z)
            self.gradient['z'] = tf.reshape(tf.reduce_mean(tf.square(dz[0]), 0), [1, -1])

    def __build_summary(self):
        with tf.name_scope('summary'):
            tf.summary.scalar('kl_loss', self.kl_loss)
            tf.summary.scalar('gen_loss', self.gen_loss)
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('gamma', self.gamma)
            self.sd_z_mean = tf.reshape(tf.reduce_mean(self.sd_z, 0), [1, -1])
            if self.decoder_num == 0:
                self.wo_norm = tf.reshape(tf.reduce_sum(tf.square(self.weights_dec['x_hat_w']), 1), [1, -1])
            else:
                self.wo_norm = tf.reshape(tf.reduce_sum(tf.square(self.weights_dec['layer0_w']), 1), [1, -1])
            self.summary = tf.summary.merge_all()

    def __build_optimizer(self):
        with tf.name_scope('optimizer'):
            with tf.name_scope('optimizer_moments'):
                self.lr = tf.placeholder(tf.float32, [], 'lr')
                self.global_step = tf.get_variable(
                    'global_step', [], tf.int32, tf.zeros_initializer(), trainable=False)
                self.optimizer = tf.train.AdamOptimizer(
                    self.lr).minimize(self.loss, self.global_step)


def write_to_file(x, filename):
    fid = open(filename, 'wb')
    pickle.dump(x, fid)
    fid.close()


def main(num_layer):
    x_dim = 400
    z_dim = 20
    kappa = 30
    encoder_dim = np.ones([num_layer], np.int32) * 200
    decoder_dim = np.ones([num_layer], np.int32) * 200
    sample_num = 20
    epoch_num = 100
    learning_rate = 0.001

    model = VaeManifold(x_dim, z_dim, kappa, encoder_dim,
                        decoder_dim, sample_num)

    fid = open('../x.bin', 'rb')
    x = pickle.load(fid)
    fid.close()

    dh_d = []
    dw_o = []
    dz = []
    w_o_norm = []
    sd_z_mean = []
    loss = []
    with tf.Session() as sess:
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter('graph')
        if not os.path.exists('model'):
            os.mkdir('model')

        sess.run(tf.global_variables_initializer())
        for epoch in range(epoch_num):
            total_loss = 0
            iteration_per_epoch = int(x.shape[0]/100)
            for i in range(iteration_per_epoch):
                gradient, batch_w_o_norm, batch_sd_z_mean, batch_loss, _, summary = sess.run(
                    [model.gradient, model.wo_norm, model.sd_z_mean, model.loss, model.optimizer, model.summary], 
                    feed_dict={model.lr: learning_rate, model.x: x[i*100:(i+1)*100, :]})
                writer.add_summary(summary, model.global_step.eval(sess))
                total_loss += batch_loss

                if i % 10 == 0:
                    dh_d.append(gradient['h_d'])
                    dw_o.append(gradient['w_o'])
                    dz.append(gradient['z'])
                    w_o_norm.append(batch_w_o_norm)
                    sd_z_mean.append(batch_sd_z_mean)
                    loss.append(batch_loss)
            total_loss /= iteration_per_epoch
            print('Epoch = {0}, loss = {1}.'.format(epoch, total_loss))

        saver.save(sess, 'model/model.ckpt')

    dw_o = np.concatenate(dw_o, 0)
    dz = np.concatenate(dz, 0)
    w_o_norm = np.concatenate(w_o_norm, 0)
    sd_z_mean = np.concatenate(sd_z_mean, 0)

    scipy.io.savemat('data.mat', {'dh_d': dh_d, 'dw_o': dw_o, 'dz': dz, 'w_o': w_o_norm, 'sd_z': sd_z_mean, 'loss': loss})


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[2]
    main(int(sys.argv[1]))

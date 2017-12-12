import tensorflow as tf
import os
import numpy as np
import pickle


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
            self.wo_norm = tf.reshape(tf.reduce_sum(tf.square(self.weights_dec['layer0_w']), 1), [1, -1])
#            self.sd_z_mean_split = tf.split(self.sd_z_mean, self.kappa)
#            w_column = tf.split(self.weights_dec['layer0_w'], self.kappa)
#            self.w_norm = []
#            for i in range(self.kappa):
#                self.w_norm.append(tf.reduce_sum(tf.square(w_column[i])))
#                tf.summary.scalar('w' + str(i), self.w_norm[i])
#                tf.summary.scalar('sd' + str(i), tf.reshape(self.sd_z_mean_split[i], []))
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


def main():
    x_dim = 400
    z_dim = 20
    kappa = 30
    encoder_dim = [200, 200, 200]
    decoder_dim = [200, 200, 200]
    sample_num = 20
    iteration_num = 100000
    learning_rate = 0.001

    model = VaeManifold(x_dim, z_dim, kappa, encoder_dim,
                        decoder_dim, sample_num)

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
        for i in range(iteration_num):
            gradient, batch_w_o_norm, batch_sd_z_mean, batch_loss, _, summary = sess.run(
                [model.gradient, model.wo_norm, model.sd_z_mean, model.loss, model.optimizer, model.summary], feed_dict={model.lr: learning_rate})
            writer.add_summary(summary, model.global_step.eval(sess))
            if i % 100 == 99:
                print('Iter = {0}, loss = {1}.'.format(i, batch_loss))

            if i % 10 == 0:
                dh_d.append(gradient['h_d'])
                dw_o.append(gradient['w_o'])
                dz.append(gradient['z'])
                w_o_norm.append(batch_w_o_norm)
                sd_z_mean.append(batch_sd_z_mean)
                loss.append(batch_loss)

        saver.save(sess, 'model/model.ckpt')

    dw_o = np.concatenate(dw_o, 0)
    dz = np.concatenate(dz, 0)
    w_o_norm = np.concatenate(w_o_norm, 0)
    sd_z_mean = np.concatenate(sd_z_mean, 0)

    write_to_file(dh_d, 'dh_d.bin')
    write_to_file(dw_o, 'dw_o.bin')
    write_to_file(dz, 'dz.bin')
    write_to_file(w_o_norm, 'w_o_norm.bin')
    write_to_file(sd_z_mean, 'sd_z_mean.bin')
    write_to_file(loss, 'loss.bin')


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()

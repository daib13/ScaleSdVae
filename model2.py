from encoder import encoder_googlenet
from decoder import decoder
import tensorflow as tf
from tensorflow.contrib import layers


def gaussian_sample(name, mu, sd):
    with tf.name_scope(name):
        epsilon = tf.random_normal(tf.shape(mu), name='epsilon')
        z = tf.add(tf.multiply(sd, epsilon), mu, 'z')
    return z


def calc_gen_loss(name, x, x_hat, gamma, log_gamma):
    with tf.name_scope(name):
        l2_distance = tf.square(x - x_hat)
        gen_loss = tf.add(l2_distance / gamma / 2.0, log_gamma, 'gen_loss')
    return l2_distance, gen_loss


def calc_kl_loss(name, mu_z, logsd_z, sd_z):
    with tf.name_scope(name):
        kl_loss = tf.divide(tf.square(mu_z) + tf.square(sd_z) - 2 * logsd_z - 1, 2.0, 'kl_loss')
    return kl_loss


class VaeNet:
    def __init__(self, variational=True, latent_dim=256, shortcut=False, weight_decay=0.00001, init_log_gamma=0.0, log_gamma_trainable=True, batch_size=32, is_train=True):
        self.variational = variational
        self.latent_dim = latent_dim
        self.shortcut = shortcut
        self.weight_decay = weight_decay
        if weight_decay > 0:
            self.reg = layers.l2_regularizer(self.weight_decay)
        else:
            self.reg = None
        self.is_train = is_train

        with tf.name_scope('input'):
            self.x = tf.placeholder(tf.float32, [batch_size, 128, 128, 3], 'x')
            self.batch_size = tf.cast(tf.shape(self.x, out_type=tf.int32)[0], tf.float32, 'batch_size')
            self.phase = tf.placeholder(tf.bool, [], 'phase')
        self.mu_z, self.logsd_z, self.sd_z = encoder_googlenet('encoder', self.x, self.phase, latent_dim, self.reg)
        if self.variational:
            self.z = gaussian_sample('sample', self.mu_z, self.sd_z)
        else:
            self.z = self.mu_z
        self.x_hat = decoder('decoder', self.z, self.phase, self.shortcut, self.reg)
        with tf.name_scope('gamma'):
            self.log_gamma = tf.get_variable('log_gamma', [], tf.float32, tf.constant_initializer(init_log_gamma), trainable=log_gamma_trainable)
            self.gamma = tf.exp(self.log_gamma, 'gamma')
        with tf.name_scope('loss'):
            l2_distance, gen_loss = calc_gen_loss('gen_loss', self.x, self.x_hat, self.gamma, self.log_gamma)
            self.gen_loss = tf.divide(tf.reduce_sum(gen_loss), self.batch_size, 'gen_loss_norm')
            self.l2_distance = tf.divide(tf.reduce_sum(l2_distance), self.batch_size, 'l2_distance')
            self.loss = self.gen_loss
            if self.variational:
                self.kl_loss = tf.divide(tf.reduce_sum(calc_kl_loss('kl_loss', self.mu_z, self.logsd_z, self.sd_z)), self.batch_size, 'kl_loss_norm')
                self.loss = tf.add(self.kl_loss, self.loss, 'loss')
        with tf.name_scope('summary'):
            tf.summary.scalar('total_loss', self.loss)
            tf.summary.scalar('gen_loss', self.gen_loss)
            tf.summary.scalar('l2_distance', self.l2_distance)
            if self.variational:
                tf.summary.scalar('kl_loss', self.kl_loss)
                tf.summary.histogram('sd_z', self.sd_z)
            tf.summary.image('raw', self.x)
            tf.summary.image('recon', self.x_hat)
            tf.summary.scalar('gamma', self.gamma)
            self.summary = tf.summary.merge_all()
        with tf.name_scope('optimizer'):
            self.lr = tf.placeholder(tf.float32, [], 'lr')
            self.global_step = tf.get_variable('global_step', [], tf.float32, tf.zeros_initializer(), trainable=False)
            self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss, self.global_step)

    def partial_train(self, x, lr, sess, writer, record=True):
        mu_z, logsd_z, sd_z, loss, _, summary = sess.run([self.mu_z, self.logsd_z, self.sd_z, self.loss, self.optimizer, self.summary], feed_dict={self.x: x, self.lr: lr, self.phase: self.is_train})
        if record:
            writer.add_summary(summary, self.global_step.eval(sess))
        return loss

    def set_phase(self, phase):
        if phase == 'TRAIN':
            self.is_train = True
        else:
            self.is_train = False
import tensorflow as tf 
from tensorflow.contrib import layers 


HALF_LOG_TWO_PI = 0.918938


class VaeModel:
    def __init__(self, num_hidden, sample_num, shortcut=False, dense=False, continuous=False):
        self.x = tf.placeholder(tf.float32, [None, 784], 'x')
        self.batch_size = tf.shape(self.x)[0]
        if not continuous:
            x_ref = tf.random_uniform([self.batch_size, 784], 0.0, 1.0)
            self.input = tf.cast(tf.greater_equal(self.x, x_ref), tf.float32, 'input')
        else:
            self.input = self.x

        with tf.name_scope('encoder'):
            with tf.variable_scope('encoder_w'):
                d_in = 784
                t_in = self.input
                for i in range(num_hidden):
                    with tf.name_scope('layer' + str(i)):
                        with tf.variable_scope('layer' + str(i) + '_w'):
                            w = tf.get_variable('w', [d_in, 200], tf.float32, layers.xavier_initializer())
                            b = tf.get_variable('b', [200], tf.float32, tf.zeros_initializer())
                            t_in = tf.nn.tanh(tf.matmul(t_in, w) + b) 
                            d_in = 200
                
        with tf.name_scope('latent'):
            with tf.variable_scope('latent'):
                w = tf.get_variable('mu_z_w', [d_in, 50], tf.float32, layers.xavier_initializer())
                b = tf.get_variable('mu_z_b', [50], tf.float32, tf.zeros_initializer())
                self.mu_z = tf.matmul(t_in, w) + b

                w = tf.get_variable('logsd_z_w', [d_in, 50], tf.float32, layers.xavier_initializer())
                b = tf.get_variable('logsd_z_b', [50], tf.float32, tf.zeros_initializer())
                self.logsd_z = tf.matmul(t_in, w) + b
                self.sd_z = tf.exp(self.logsd_z)

                tile_mu_z = tf.tile(self.mu_z, [sample_num, 1])
                tile_logsd_z = tf.tile(self.logsd_z, [sample_num, 1])
                tile_sd_z = tf.tile(self.sd_z, [sample_num, 1])

                noise = tf.random_normal([self.batch_size*sample_num, 50])
                self.z = noise * tile_sd_z + tile_mu_z 

        self.wo = None
        with tf.name_scope('decoder'):
            with tf.variable_scope('decoder_w'):
                d_in = 50
                t_in = self.z 
                for i in range(num_hidden):
                    with tf.name_scope('layer' + str(i)):
                        with tf.variable_scope('layer' + str(i) + '_w'):
                            w = tf.get_variable('w', [d_in, 200], tf.float32, layers.xavier_initializer())
                            b = tf.get_variable('b', [200], tf.float32, tf.zeros_initializer())
                            t_out = tf.nn.tanh(tf.matmul(t_in, w) + b)
                            t_in = t_out if not dense else tf.concat([t_in, t_out], -1)
                            d_in = 200 if not dense else d_in + 200
                            if self.wo is None:
                                self.wo = w 
        
        with tf.name_scope('recon'):
            with tf.variable_scope('recon_w'):
                if shortcut:
                    t_in = tf.concat([t_in, self.z], -1)
                    d_in += 50
                w = tf.get_variable('w', [d_in, 784], tf.float32, layers.xavier_initializer())
                b = tf.get_variable('b', [784], tf.float32, tf.zeros_initializer())
                self.x_hat_logit = tf.matmul(t_in, w) + b
                self.x_hat = tf.nn.sigmoid(self.x_hat_logit)
                if self.wo is None:
                    self.wo = w

        with tf.name_scope('loss'):
            self.kl_coef = tf.placeholder(tf.float32, [], 'kl_coef')
            self.kl_loss = tf.reduce_sum(tf.square(self.mu_z) + tf.square(self.sd_z) - 2 * self.logsd_z - 1) / 2.0 / tf.cast(self.batch_size, tf.float32)
            tile_input = tf.tile(self.input, [sample_num, 1])
            if not continuous:
                self.gen_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=tile_input, logits=self.x_hat_logit)) / tf.cast(self.batch_size, tf.float32) / sample_num
                self.gamma = None
            else:
                self.loggamma = tf.get_variable('loggamma', [], tf.float32, tf.constant_initializer(0))
                self.gamma = tf.exp(self.loggamma, 'gamma')
                self.gen_loss = tf.reduce_sum(tf.square((self.x_hat - self.input) / self.gamma) / 2.0 + self.loggamma + HALF_LOG_TWO_PI) / tf.cast(self.batch_size, tf.float32)
            self.loss = self.kl_coef * self.kl_loss + self.gen_loss 

        with tf.name_scope('summary'):
            tf.summary.scalar('kl_loss', self.kl_loss)
            tf.summary.scalar('gen_loss', self.gen_loss)
            tf.summary.scalar('loss', self.loss)
            if self.gamma is not None:
                tf.summary.scalar('gamma', self.gamma)
            tf.summary.image('raw', tf.reshape(self.input, [self.batch_size, 28, 28, 1]))
            tf.summary.image('recon', tf.reshape(self.x_hat, [self.batch_size, 28, 28, 1]))
            self.summary = tf.summary.merge_all() 

        with tf.name_scope('nll'):
            kl_logit = tf.reduce_sum((tf.square(noise) - tf.square(self.z)) / 2.0, -1)
            if not continuous:
                gen_logit = - tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=tile_input, logits=self.x_hat_logit), -1) 
            else:
                gen_logit = - tf.reduce_sum(tf.square((self.x_hat - tile_input) / self.gamma) / 2.0 + self.loggamma + HALF_LOG_TWO_PI, -1)
            logit = kl_logit + gen_logit
            logit = -tf.reshape(logit, [sample_num, self.batch_size])
            logit_max = tf.reduce_max(logit, 0)
            logit_residual = logit - tf.tile(tf.reshape(logit_max, [1, -1]), [sample_num, 1])
            nll = logit_max + tf.log(tf.reduce_sum(tf.exp(logit_residual), 0))
            self.nll = tf.reduce_mean(nll)


class VaeOptimizer:
    def __init__(self, model):
        with tf.name_scope('optimizer'):
            with tf.variable_scope('optimizer_w'):
                self.model = model
                self.lr = tf.placeholder(tf.float32, [], 'learning_rate')
                self.global_step = tf.get_variable('global_step', [], tf.int32, tf.zeros_initializer(), trainable=False)
                self.optimizer = tf.train.AdamOptimizer(self.lr, epsilon=0.0001).minimize(model.loss, self.global_step)

    def optimize(self, sess, x, lr, kl_coef=1.0):
        loss, _, summary = sess.run([self.model.loss, self.optimizer, self.model.summary], feed_dict={self.model.x: x, self.model.kl_coef: kl_coef, self.lr: lr})
        return loss, summary 
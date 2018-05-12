import tensorflow as tf 
from tensorflow.contrib import layers 


class VaeModel:
    def __init__(self, num_hidden, sample_num, shortcut=False, dense=False):
        self.x = tf.placeholder(tf.float32, [None, 784], 'x')
        self.batch_size = tf.shape(self.x)[0]
        x_ref = tf.random_normal([self.batch_size, 784])
        self.input = tf.greater_equal(self.x, x_ref, 'input')

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
            self.kl_loss = tf.reduce_sum(tf.square(self.mu_z) + tf.square(self.sd_z) - 2 * self.logsd_z - 1) / 2.0 / tf.cast(self.batch_size, tf.int32)
            self.gen_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input, logits=self.x_hat_logit)) / tf.cast(self.batch_size, tf.int32)
            self.loss = self.kl_coef * self.kl_loss + self.gen_loss 

        with tf.name_scope('summary'):
            tf.summary.scalar('kl_loss', self.kl_loss)
            tf.summary.scalar('gen_loss', self.gen_loss)
            tf.summary.scalar('loss', self.loss)
            self.summary = tf.summary.merge_all() 

        with tf.name_scope('nll'):
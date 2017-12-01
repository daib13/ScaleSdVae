from nn import res_block
import tensorflow as tf
from tensorflow.contrib import layers


class model:
    def __init__(self, num_block, num_layer_per_block, num_filter, filter_size=[3, 3], padding='SAME',
                 fc_dim=[1024], latent_dim=64, weight_decay=0, activation_fn=tf.nn.elu):
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

        self.__build_net()

    def __build_net(self):
        self.__build_encoder()

    def __build_encoder(self):
        with tf.name_scope('x'):
            self.x = tf.placeholder(tf.float32, self.img_shape)
        feature_map = self.x
        with tf.name_scope('encoder'):
            with tf.variable_scope('encoder_w'):
                for i_block in range(num_block):
                    with tf.name_scope('block' + str(i_block)):
                        feature_map = res_block('block' + str(i_block) + 'w', feature_map, self.num_layer_per_block,
                                                self.num_filter[i_block], self.filter_size, self.padding, self.reg, self.activation_fn)
                    with tf.name_scope('pool' + str(i_block)):
                        feature_map = tf.nn.max_pool(feature_map, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
                feature = tf.reshape(feature_map, [int(feature_map.get_shape()[0]), -1])
                for i_fc in range(self.num_fc_layer):
                    

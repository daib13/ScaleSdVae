from nn import VaeSimple, VaeScale
import data_util as du 
import os
from mnist import MNIST
import math
import numpy as np
import tensorflow as tf
import sys


def train_model(model, x, lr, num_epoch, batch_size, sess, writer):
    iteration_per_epoch = int(math.floor(x.shape[0] / batch_size))
    for epoch in range(num_epoch):
        x = du.shuffle_data(x)
        total_loss = 0
        for i in range(iteration_per_epoch):
            batch_x = x[i*batch_size:(i+1)*batch_size, :]
            batch_loss = model.partial_train(batch_x, lr, sess, writer)
            total_loss += batch_loss
        total_loss /= iteration_per_epoch
        print('Lr = {0}, epoch = {1}, loss = {2}.'.format(lr, epoch, total_loss))


def test_model(model, x, batch_size, sess):
    num_batch = int(math.ceil(x.shape[0] / batch_size))
    nll = []
    for batch in range(num_batch):
        min_id = batch*batch_size
        max_id = min((batch+1)*batch_size, x.shape[0])
        batch_nll = model.test_nll(x[min_id:max_id], sess)
        nll.append(batch_nll)
    nll = np.concatenate(nll, 0)
    nll = nll[0:x.shape[0]]
    return np.mean(nll)


def main(num_layer, gamma):
    x_train, y_train = du.load_mnist_data('training')
    x_test, y_test = du.load_mnist_data('testing')

    input_dim = 784
    latent_dim = 50
    encoder_dim = np.ones([num_layer], np.int32) * 200
    decoder_dim = np.ones([num_layer], np.int32) * 200
    sample_num = 1
    batch_size = 20
    stage_num = 8

#    model = VaeSimple(input_dim, latent_dim, encoder_dim, decoder_dim, 5000)
    model = VaeScale(input_dim, latent_dim, encoder_dim, decoder_dim, 5000, gamma=gamma)
    with tf.Session() as sess:
        saver = tf.train.Saver()
        model.restore_from(saver, sess, 'model/stage' + str(stage_num - 1))

        nll = test_model(model, x_test, batch_size, sess)
        print('NLL = {0}.'.format(nll))


if __name__ == '__main__':
    num_layer = int(sys.argv[1])
    gamma = float(sys.argv[2])
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[3]
    print('num_layer = {0}, gamma = {1}, gpu_id = {2}.'.format(num_layer, gamma, sys.argv[3]))
    main(num_layer, gamma)


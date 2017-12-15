from dataset import make_cifar10_dataset, shuffle_data
from model import ConvVae, ConvAe
import tensorflow as tf
import math
import os
from config import config
import sys


def train_model(config_name, model_name, num_epoch=100, batch_size=16, lr=0.0001):
    x_train, y_train, x_test, y_test = make_cifar10_dataset()

    margs = config(config_name)
    num_block = margs['num_block']
    num_layer_per_block = margs['num_layer_per_block']
    num_filter = margs['num_filter']
    fc_dim = margs['fc_dim']
    latent_dim = margs['latent_dim']
    if model_name == 'ConvAe':
        model = ConvAe(num_block, num_layer_per_block, num_filter, fc_dim=fc_dim, latent_dim=latent_dim)
    elif model_name == 'ConvVae':
        model = ConvVae(num_block, num_layer_per_block, num_filter, fc_dim=fc_dim, latent_dim=latent_dim)
    else:
        print('No model named {0}.'.format(model_name))
        return

    with tf.Session() as sess:
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter('graph', sess.graph)

        sess.run(tf.global_variables_initializer())    

        iteration_per_epoch = int(math.floor(x_train.shape[0] / batch_size))
        for epoch in range(num_epoch):
            shuffle_data(x_train)
            total_loss = 0
            for i in range(iteration_per_epoch):
                x_batch = x_train[i*batch_size:(i+1)*batch_size, :, :, :]
                loss, _, summary = sess.run([model.loss, model.optimizer, model.summary], feed_dict={model.x: x_batch, model.lr: lr})
                total_loss += loss
                writer.add_summary(summary, model.global_step.eval(sess))
            total_loss /= iteration_per_epoch
            print('Epoch = {0}, loss = {1}.'.format(epoch, total_loss))

        saver.save(sess, 'model/model.ckpt')


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[3]
    train_model(sys.argv[1], sys.argv[2])
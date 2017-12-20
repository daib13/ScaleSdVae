from model2 import VaeNet
import tensorflow as tf 
from dataset import load_img_from_folder, shuffle_data
import math
import os
import sys
from config import data_folder


def train_model(model, sess, writer, x, num_epoch, lr, batch_size=32):
    iteration_per_epoch = int(math.floor(x.shape[0] / batch_size))
    total_iter = 0
    for epoch in range(num_epoch):
        x = shuffle_data(x)
        total_loss = 0
        for i in range(iteration_per_epoch):
            x_batch = x[i*batch_size:(i+1)*batch_size, :, :, :]
            total_iter += 1
            batch_loss = model.partial_train(x_batch, lr, sess, writer, total_iter % 10 == 0)
            total_loss += batch_loss
            print('Iter = {0}, loss = {1}.'.format(total_iter, batch_loss))
        total_loss /= iteration_per_epoch
        print('Epoch = {0}, lr = {1}, loss = {2}.'.format(epoch, lr, total_loss))


def main(data_set, model_type, latent_dim, shortcut='True', num_epoch=100, log_gamma_decay=0.0):
    data_dir = data_folder(data_set)
    if data_dir == '':
        print('No such data set named {0}.'.format(data_set))
        return
    x = load_img_from_folder(data_dir)

    if model_type == 'VAE':
        variational = True
    elif model_type == 'AE':
        variational = False
    else:
        print('No such model type named {0}.'.format(model_type))
        return

    if shortcut == 'True':
        resnet = True
    elif shortcut == 'False':
        resnet = False
    else:
        print('Shortcut setting unclear: {0}.'.format(shortcut))
        return

    model = VaeNet(variational, latent_dim=latent_dim, shortcut=resnet, init_log_gamma=-4.0, log_gamma_decay=log_gamma_decay)
    
    if not os.path.exists('model'):
        os.mkdir('model')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter('graph', sess.graph)

        train_model(model, sess, writer, x, num_epoch, 0.002, 32)
        saver.save(sess, 'model/model.ckpt')


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[6]
    data_set = sys.argv[1]
    model_type = sys.argv[2]
    latent_dim = int(sys.argv[3])
    log_gamma_decay = float(sys.argv[4])
    shortcut = sys.argv[5]
    main(data_set, model_type, latent_dim, shortcut, num_epoch=200, log_gamma_decay=log_gamma_decay)

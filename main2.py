from model2 import VaeNet
import tensorflow as tf 
from dataset import load_img_from_folder, shuffle_data
import math
import os
import sys
from config import data_folder


def train_model(model, sess, writer, x, num_epoch, lr, batch_size=32, decay=False):
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
        if decay and (epoch % 20 == 19):
            lr *= 0.1
        total_loss /= iteration_per_epoch
        print('Epoch = {0}, lr = {1}, loss = {2}.'.format(epoch, lr, total_loss))


def main(data_set, model_type, shortcut, num_epoch=100, lr=0.002, init_log_gamma=-4.0, decay=False):
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

    model = VaeNet(variational, shortcut=resnet, init_log_gamma=init_log_gamma, log_gamma_trainable=True)
    
    if not os.path.exists('model'):
        os.mkdir('model')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter('graph', sess.graph)

        train_model(model, sess, writer, x, num_epoch, lr, 32, decay)
        saver.save(sess, 'model/model.ckpt')


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[6]
    data_set = sys.argv[1]
    model_type = sys.argv[2]
    shortcut = sys.argv[3]
    num_epoch = int(sys.argv[4])
    lr = 0.002
    init_log_gamma = float(sys.argv[5])
    main(data_set, model_type, shortcut, num_epoch, lr, init_log_gamma, True)

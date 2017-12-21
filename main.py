from model2 import VaeNet
import tensorflow as tf 
from dataset import load_img_from_folder, shuffle_data
import math
import os
import sys
from config import data_folder
import numpy as np 
import matplotlib.image as mpimg
from beta_policy import constant_beta, linear_beta


def train_model(model, sess, writer, x, num_epoch, lr, batch_size=32, policy_type=constant_beta):
    iteration_per_epoch = int(math.floor(x.shape[0] / batch_size))
    total_iter = 0
    for epoch in range(num_epoch):
        x = shuffle_data(x)
        beta = policy_type(epoch, num_epoch)
        total_loss = 0
        for i in range(iteration_per_epoch):
            x_batch = x[i*batch_size:(i+1)*batch_size, :, :, :]
            total_iter += 1
            batch_loss = model.partial_train(x_batch, lr, sess, writer, total_iter % 10 == 0, beta)
            total_loss += batch_loss
        total_loss /= iteration_per_epoch
        print('Epoch = {0}, beta = {1}, lr = {2}, loss = {3}.'.format(epoch, beta, lr, total_loss))


def generate_sample(model, sess, num_sample):
    total_iteration = int(math.ceil(num_sample / 10))
    x = []
    for i in range(total_iteration):
        batch_x = model.generate_sample(sess)
        x.append(batch_x)
    x = np.concatenate(x, 0)
    return x


def main(data_set, model_type='VAE', latent_dim=256, shortcut='True', strategy='normal', num_epoch=100, log_gamma_decay=0.0, output_fn_name='sigmoid', init_log_gamma=-4.0):
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

    if strategy == 'normal':
        beta_policy = constant_beta
        scale_std = False
    elif strategy == 'warmup':
        beta_policy = linear_beta
        scale_std = False
    elif strategy == 'scale':
        beta_policy = constant_beta
        scale_std = True
    else:
        print('No strategy named {0}.'.format(strategy))
        return

    if output_fn_name == 'sigmoid':
        output_fn = tf.nn.sigmoid
    elif output_fn_name == 'none':
        output_fn = None
    else:
        print('No output function named {0}.'.format(output_fn_name))
        return

    model = VaeNet(variational, latent_dim=latent_dim, shortcut=resnet, init_log_gamma=init_log_gamma, scale_std=scale_std, log_gamma_trainable=False, output_fn=output_fn)
    
    if not os.path.exists('model'):
        os.mkdir('model')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter('graph', sess.graph)

        train_model(model, sess, writer, x, num_epoch, 0.002, 32, beta_policy)
        saver.save(sess, 'model/model.ckpt')

    tf.reset_default_graph()
    model = VaeNet(variational, latent_dim=latent_dim, shortcut=resnet, is_train=False, scale_std=scale_std, output_fn=output_fn)
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, 'model/model.ckpt')
        samples = generate_sample(model, sess, 500)
        if not os.path.exists('samples'):
            os.mkdir('samples')
        for i in range(500):
            mpimg.imsave('samples/' + str(i) + '.jpg', samples[i, :, :, :], 0.0, 1.0)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[6]
    data_set = sys.argv[1]
    shortcut = sys.argv[2]
    strategy = sys.argv[3]
    output_fn = sys.argv[4]
    init_log_gamma = float(sys.argv[5])
    main(data_set, shortcut=shortcut, strategy=strategy, num_epoch=300, output_fn_name=output_fn, init_log_gamma=init_log_gamma)

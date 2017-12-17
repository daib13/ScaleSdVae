from model2 import VaeNet
import tensorflow as tf 
from dataset import load_img_from_folder, shuffle_data
import math
import os


def train_model(model, sess, writer, x, num_epoch, lr, batch_size=32):
    iteration_per_epoch = int(math.floor(x.shape[0] / batch_size))
    for epoch in range(num_epoch):
        x = shuffle_data(x)
        total_loss = 0
        for i in range(iteration_per_epoch):
            x_batch = x[i*batch_size:(i+1)*batch_size, :, :, :]
            batch_loss = model.partial_train(x_batch, lr, sess, writer, True)
            total_loss += batch_loss
        total_loss /= iteration_per_epoch
        print('Epoch = {0}, lr = {1}, loss = {2}.'.format(epoch, lr, total_loss))


def main():
    data_folder = 'F:/Projects/VAEDifficulty/test_imgs'
    x = load_img_from_folder(data_folder)

    model = VaeNet(True, batch_size=10, shortcut=True)
    
    if not os.path.exists('model'):
        os.mkdir('model')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter('graph', sess.graph)

        train_model(model, sess, writer, x, 1000, 0.003, 10)
        saver.save(sess, 'model/model.ckpt')


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()
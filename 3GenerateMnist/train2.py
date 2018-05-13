import argparse
from nn2 import VaeModel, VaeOptimizer
import os 
import tensorflow as tf 
from data_util import load_mnist_data, shuffle_data
import math 
import time 
import numpy as np 


def main():
    
    # exp info
    save_dir = os.path.join('experiments', 'save')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    exp_id = args.exp_id if args.exp_id != '' else str(len(os.listdir(save_dir)))
    save_path = os.path.join(save_dir, exp_id)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    summary_dir = os.path.join('experiments', 'summary')
    if not os.path.exists(summary_dir):
        os.mkdir(summary_dir)
    summary_path = os.path.join(summary_dir, exp_id)
    if not os.path.exists(summary_path):
        os.mkdir(summary_path)

    # prepare data
    x_train, y_train = load_mnist_data('training')
    x_test, y_test = load_mnist_data('testing')
    print(np.shape(x_train))
    print(np.shape(x_test))

    # model
    model = VaeModel(args.num_hidden, 1, args.shortcut, args.dense, args.continuous)
    optimizer = VaeOptimizer(model)

    # train
    if not args.val:
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            writer = tf.summary.FileWriter(summary_path, sess.graph)
            saver = tf.train.Saver()

            for stage in range(args.num_stage):
                lr = 0.001 * math.pow(0.1, float(stage)/float(args.num_stage-1))
                epochs = math.pow(3, stage)
                kl_coef = 0.001 * math.pow(1000, float(stage)/float(args.num_stage-1)) if args.warmup else 1.0
                print('Stage = {}, lr = {:.6f}, kl_coef = {:.4f} epochs = {}.'.format(stage, lr, kl_coef, epochs))
                train(optimizer, sess, x_train, epochs, lr, kl_coef, writer)
                saver.save(sess, os.path.join(save_path, str(stage)))

    tf.reset_default_graph()
    model = VaeModel(args.num_hidden, 5000, args.shortcut, args.dense, args.continuous)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, os.path.join(save_path, str(args.num_stage-1)))
        nll = test_nll(model, sess, x_test)
        print('NLL = {:.4f}.'.format(nll))

        wo = model.wo.eval(sess)
        wo_norm = np.sum(np.square(wo), -1)
        wo_norm = -np.sort(-wo_norm)
        fid = open(os.path.join(save_path, 'wo.txt'), 'wt')
        for w in wo_norm:
            fid.write('{:.4f}\n'.format(w))
        fid.close()
        print('Num active: {}.'.format(np.sum(wo_norm > 0.05 * np.max(wo_norm))))


def train(optimizer, sess, x, epochs, lr, kl_coef, writer=None):
    iteration_per_epoch = math.floor(np.shape(x)[0] / args.batch_size)
    for epoch in range(int(epochs)):
        loss = 0
        x = shuffle_data(x)
        for i in range(iteration_per_epoch):
            x_batch = x[i*args.batch_size:(i+1)*args.batch_size]
            batch_loss, batch_summary = optimizer.optimize(sess, x_batch, lr, kl_coef)
            loss += batch_loss
        loss /= iteration_per_epoch
        if writer is not None:
            writer.add_summary(batch_summary, optimizer.global_step.eval(sess))
        print('Date: {date}\t'
              'Epoch: [{0}/{1}]\t'
              'Loss: {2:.4f}.'.format(epoch, epochs, loss, date=time.strftime('%Y-%m-%d %H:%M:%S')))


def test_nll(model, sess, x):
    num_iteration = math.floor(np.shape(x)[0] / args.batch_size)
    nll = 0
    for i in range(num_iteration):
        x_batch = x[i*args.batch_size:(i+1)*args.batch_size]
        batch_nll = sess.run(model.nll, feed_dict={model.x: x_batch})
        nll += batch_nll 
    nll /= num_iteration
    return nll


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp-id', type=str, default='debug')
    parser.add_argument('--num-hidden', type=int, default=2)
    parser.add_argument('--shortcut', default=False, action='store_true')
    parser.add_argument('--dense', default=False, action='store_true')
    parser.add_argument('--warmup', default=False, action='store_true')
    parser.add_argument('--batch-size', type=int, default=20)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--val', default=False, action='store_true')
    parser.add_argument('--num-stage', type=int, default=8)
    parser.add_argument('--continuous', default=False, action='store_true')

    args = parser.parse_args()
    print(args)

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    main()
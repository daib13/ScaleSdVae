import numpy as np
import os
import pickle
from os import listdir
from os.path import isfile, join
import matplotlib.image as mpimg


def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='bytes')
        x = datadict[b'data']
        y = datadict[b'labels']
        x = x.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
        y = np.array(y)
        return x, y


def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        x, y = load_CIFAR_batch(f)
        xs.append(x)
        ys.append(y)
    xtr = np.concatenate(xs)
    ytr = np.concatenate(ys)
    del x, y
    xte, yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return xtr / 255.0, ytr, xte / 255.0, yte


def make_cifar10_dataset(vectorize=False):
    num_classes = 10
    num_train   = 50000
    num_test    = 10000

    cifar10_dir = 'D:/v-bindai/Projects/VAEDifficulty/data/cifar-10-batches-py'
    x_train, y_train, x_test, y_test = load_CIFAR10(cifar10_dir)

    # reshape to vectors
    if vectorize:
        x_train = np.reshape(x_train,(x_train.shape[0],-1))
        x_test  = np.reshape(x_test,(x_test.shape[0],-1))

    # make one-hot coding
    y_train_temp = np.zeros((num_train,num_classes))
    for i in range(num_train):
        y_train_temp[i,y_train[i]] = 1
    y_train = y_train_temp

    y_test_temp = np.zeros((num_test,num_classes))
    for i in range(num_test):
        y_test_temp[i,y_test[i]] = 1
    y_test = y_test_temp

    return x_train, y_train, x_test, y_test


def shuffle_data(x, y=None):
    idx = np.arange(x.shape[0])
    np.random.shuffle(idx)
    x = x[idx, :]
    if y is not None:
        y = y[idx]
        return x, y
    else:
        return x


def load_img_from_folder(folder_name, height=128, width=128, channel=3, num_img=-1):
    x = []
    finish = False
    for dirpath, dirnames, files in os.walk(folder_name):
        for name in files:
            if name.lower().endswith('.jpg'):
                img_file = os.path.join(dirpath, name)
                img = mpimg.imread(img_file)
                assert len(img.shape) == 3 or len(img.shape) == 2
                if len(img.shape) == 2:
                    img = np.tile(np.reshape(img, [height, width, 1]), [1, 1, 3])
                assert img.shape[0] == height
                assert img.shape[1] == width
                assert img.shape[2] == channel
                x.append(np.reshape(img, [1, height, width, channel]))
                if len(x) % 5000 == 0:
                    print('loading image {0}...'.format(len(x)))
                if num_img > 0 and len(x) >= num_img:
                    finish = True
                    print('Finish loading {0} images.'.format(num_img))
                    break
        if finish:
            break
    x = np.concatenate(x, 0) / 255.0
    return x

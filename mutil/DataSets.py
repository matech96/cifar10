from collections import namedtuple

import keras
import numpy as np
from keras.datasets import cifar10
from sklearn.utils import shuffle

DataSets = namedtuple('DataSets', 'x_dev x_test x_train y_dev y_test y_train')


def get_cifar10_data(data_portion: float = 1.0) -> DataSets:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    X = np.concatenate((x_train, x_test), axis=0)
    Y = np.concatenate((y_train, y_test), axis=0)
    Y = keras.utils.to_categorical(Y, 10)
    n_data_points = X.shape[0]
    random_indecies = shuffle(np.arange(n_data_points), random_state=42)
    X = X[random_indecies,]
    Y = Y[random_indecies,]
    border_train = int(n_data_points * 0.7)
    border_dev = int(n_data_points * 0.9)
    x_train = X[:int(border_train*data_portion), ]
    y_train = Y[:int(border_train*data_portion), ]
    x_dev = X[border_train:border_dev, ]
    y_dev = Y[border_train:border_dev, ]
    x_test = X[border_dev:, ]
    y_test = Y[border_dev:, ]
    return DataSets(x_dev, x_test, x_train, y_dev, y_test, y_train)
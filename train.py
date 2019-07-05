import os
from sklearn.utils import shuffle

from comet_ml import Experiment
import keras
from keras import Sequential
from keras import regularizers
from keras.callbacks import CSVLogger, EarlyStopping
from keras.datasets import cifar10
from keras.layers import Conv2D, Activation, MaxPooling2D, Dropout, Flatten, Dense
from keras.optimizers import Adam
from keras.utils import plot_model
import numpy as np
from typing import Callable


def get_model(input_shape: np.ndarray, num_classes: int, regularization: regularizers.Regularizer):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), kernel_regularizer=regularization))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularization))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), kernel_regularizer=regularization))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512, kernel_regularizer=regularization))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, kernel_regularizer=regularization))
    model.add(Activation('softmax'))
    return model


def get_model_same(input_shape: np.ndarray, num_classes: int, regularization: regularizers.Regularizer):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularization))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularization))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularization))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512, kernel_regularizer=regularization))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, kernel_regularizer=regularization))
    model.add(Activation('softmax'))
    return model


def get_model_3block(input_shape: np.ndarray, num_classes: int, regularization: regularizers.Regularizer):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularization))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularization))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, kernel_regularizer=regularization))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, kernel_regularizer=regularization))
    model.add(Activation('softmax'))
    return model


def get_model_4block(input_shape: np.ndarray, num_classes: int, regularization: regularizers.Regularizer):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularization))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularization))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularization))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, kernel_regularizer=regularization))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, kernel_regularizer=regularization))
    model.add(Activation('softmax'))
    return model


def get_model_5block(input_shape: np.ndarray, num_classes: int, regularization: regularizers.Regularizer):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularization))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularization))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularization))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularization))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, kernel_regularizer=regularization))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, kernel_regularizer=regularization))
    model.add(Activation('softmax'))
    return model


def train_cifar10(batch_size: int, learning_rate: float, epochs: int, experiment: Experiment,
                  regularization: regularizers.Regularizer = None,
                  model_fnc: Callable[[np.ndarray, int, regularizers.Regularizer], Sequential] = get_model) -> None:
    model_plot_file_name = 'model.png'
    name = experiment.get_key()
    num_classes = 10
    save_dir = os.path.join(os.getcwd(), 'results')
    model_name = '{}.h5'.format(name)
    log_name = '{}.csv'.format(name)
    model_path = os.path.join(save_dir, model_name)
    log_path = os.path.join(save_dir, log_name)
    # The data, split between train and test sets:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    X = np.concatenate((x_train, x_test), axis=0)
    Y = np.concatenate((y_train, y_test), axis=0)
    Y = keras.utils.to_categorical(Y, num_classes)

    n_data_points = X.shape[0]
    random_indecies = shuffle(np.arange(n_data_points), random_state=42)
    X = X[random_indecies,]
    Y = Y[random_indecies,]
    X = X.astype('float32') / 255.0
    Y = Y.astype('float32')
    border_train = int(n_data_points * 0.7)
    border_dev = int(n_data_points * 0.9)
    x_train = X[:border_train, ]
    y_train = Y[:border_train, ]
    x_dev = X[border_train:border_dev, ]
    y_dev = Y[border_train:border_dev, ]
    x_test = X[border_dev:, ]
    y_test = Y[border_dev:, ]

    input_shape = x_train.shape[1:]
    model = model_fnc(input_shape, num_classes, regularization)
    opt = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    plot_model(model, model_plot_file_name, show_shapes=True)
    experiment.log_image(model_plot_file_name)
    os.remove(model_plot_file_name)
    csv_cb = CSVLogger(log_path)
    early_stopping = EarlyStopping('val_acc', patience=250, restore_best_weights=True, verbose=2)
    callbacks = [csv_cb, early_stopping]
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_dev, y_dev),
              shuffle=True,
              callbacks=callbacks,
              verbose=2)
    model.save(model_path)
    scores = model.evaluate(x_train, y_train, verbose=2)
    print('Train loss:', scores[0])
    print('Train accuracy:', scores[1])
    experiment.log_metrics({"loss": scores[0],
                            "acc": scores[1]}, prefix="train")

    experiment.log_asset(model_path)
    scores = model.evaluate(x_dev, y_dev, verbose=2)
    print('Dev loss:', scores[0])
    print('Dev accuracy:', scores[1])
    experiment.log_metrics({"loss": scores[0],
                            "acc": scores[1]}, prefix="dev")

    scores = model.evaluate(x_test, y_test, verbose=2)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
    experiment.log_metrics({"loss": scores[0],
                            "acc": scores[1]}, prefix="test")

import os

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
    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    input_shape = x_train.shape[1:]
    model = model_fnc(input_shape, num_classes, regularization)
    opt = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    plot_model(model, model_plot_file_name, show_shapes=True)
    experiment.log_asset(model_plot_file_name)
    os.remove(model_plot_file_name)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    csv_cb = CSVLogger(log_path)
    early_stopping = EarlyStopping('val_acc', patience=250, restore_best_weights=True)
    callbacks = [csv_cb, early_stopping]
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True,
              callbacks=callbacks,
              verbose=2)
    model.save(model_path)
    experiment.log_asset(model_path)
    # Score trained model.
    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

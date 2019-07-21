import os

from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import shuffle

from comet_ml import Experiment
import keras
from keras import Sequential
from keras.callbacks import CSVLogger, EarlyStopping
from keras.datasets import cifar10
from keras.optimizers import Adam
from keras.utils import plot_model
import numpy as np

from models import get_model
from mutil import ElapsedTime


def train_cifar10(batch_size: int, learning_rate: float, epochs: int, experiment: Experiment,
                  model: Sequential = get_model(), initial_epoch: int = 0,
                  training_datagen: ImageDataGenerator = ImageDataGenerator()) -> None:
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
    border_train = int(n_data_points * 0.7)
    border_dev = int(n_data_points * 0.9)
    x_train = X[:border_train, ]
    y_train = Y[:border_train, ]
    x_dev = X[border_train:border_dev, ]
    y_dev = Y[border_train:border_dev, ]
    x_test = X[border_dev:, ]
    y_test = Y[border_dev:, ]

    training_datagen.fit(x_train)
    log_images(x_train, training_datagen, experiment)
    log_input_images(x_train, y_train, training_datagen, experiment)

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

    model.fit_generator(training_datagen.flow(x_train, y_train, batch_size=batch_size),
                        steps_per_epoch=len(x_train) / batch_size,
                        epochs=epochs,
                        validation_data=(x_dev, y_dev),
                        shuffle=True,
                        callbacks=callbacks,
                        verbose=2,
                        initial_epoch=initial_epoch)
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

    timer = ElapsedTime("Test prediction")
    with timer:
        scores = model.evaluate(x_test, y_test, verbose=2)
    experiment.log_metric("test_inference_time", timer.elapsed_time_ms)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
    experiment.log_metrics({"loss": scores[0],
                            "acc": scores[1]}, prefix="test")


def log_input_images(x_train, y_train, training_datagen, experiment):
    imgs = training_datagen.flow(x_train, y_train, batch_size=10)[0][0]
    for i in range(10):
        experiment.log_image(imgs[i, :], 'smp_{}'.format(i))


def log_images(x_train, training_datagen, experiment):
    for i in range(10):
        x_i = x_train[i,]
        experiment.log_image(x_i, '{}'.format(i))
        trf = training_datagen.get_random_transform(x_train.shape[1:])
        trf_x = training_datagen.apply_transform(x_i, trf)
        experiment.log_image(trf_x, '{}_trf'.format(i))

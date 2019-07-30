from comet_ml import Experiment
import os

from keras import Sequential
from keras.callbacks import CSVLogger, EarlyStopping, LearningRateScheduler
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator

from models import get_model
from mutil import ElapsedTime
from mutil.DataSets import get_cifar10_data

from typing import Callable


def train_cifar10(batch_size: int, learning_rate: float, epochs: int, experiment: Experiment,
                  model: Sequential = get_model(), initial_epoch: int = 0,
                  training_datagen: ImageDataGenerator = ImageDataGenerator(),
                  scheduler: Callable[[int], float] = None) -> None:
    preprocessing_fnc = training_datagen.preprocessing_function
    name = experiment.get_key()
    log_path, model_path = get_output_paths(name)
    data = get_cifar10_data()

    training_datagen.fit(data.x_train)
    log_images(data.x_train, training_datagen, experiment)
    log_input_images(data.x_train, data.y_train, training_datagen, experiment)

    opt = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    log_model_plot(experiment, model)

    csv_cb = CSVLogger(log_path)
    early_stopping_cb = EarlyStopping('val_acc', patience=epochs, restore_best_weights=True, verbose=2)
    callbacks = [csv_cb, early_stopping_cb]
    if scheduler is not None:
        scheduler.experiment_log(experiment=experiment, epochs=list(range(epochs)))
        callbacks.append(LearningRateScheduler(scheduler))

    model.fit_generator(training_datagen.flow(data.x_train, data.y_train, batch_size=batch_size),
                        steps_per_epoch=len(data.x_train) / batch_size,
                        epochs=epochs,
                        validation_data=(preprocessing_fnc(data.x_dev), data.y_dev),
                        shuffle=True,
                        callbacks=callbacks,
                        verbose=2,
                        initial_epoch=initial_epoch)
    model.save(model_path)
    experiment.log_asset(model_path)
    experiment.log_asset(log_path)

    log_final_metrics(experiment, model, data, preprocessing_fnc)


def get_output_paths(name):
    save_dir = os.path.join(os.getcwd(), 'results')
    model_name = '{}.h5'.format(name)
    log_name = '{}.csv'.format(name)
    model_path = os.path.join(save_dir, model_name)
    log_path = os.path.join(save_dir, log_name)
    return log_path, model_path


def log_images(x_train, training_datagen, experiment):
    for i in range(10):
        x_i = x_train[i,]
        experiment.log_image(x_i, '{}'.format(i))
        trf = training_datagen.get_random_transform(x_train.shape[1:])
        trf_x = training_datagen.apply_transform(x_i, trf)
        experiment.log_image(trf_x, '{}_trf'.format(i))


def log_input_images(x_train, y_train, training_datagen, experiment):
    imgs = training_datagen.flow(x_train, y_train, batch_size=10)[0][0]
    for i in range(10):
        experiment.log_image(imgs[i, :], 'smp_{}'.format(i))


def log_model_plot(experiment, model):
    model_plot_file_name = 'model.png'
    plot_model(model, model_plot_file_name, show_shapes=True)
    experiment.log_image(model_plot_file_name)
    os.remove(model_plot_file_name)


def log_final_metrics(experiment, model, data, preprocessing_fnc):
    scores = model.evaluate(preprocessing_fnc(data.x_train), data.y_train, verbose=2)
    experiment.log_metrics({"loss": scores[0],
                            "acc": scores[1]}, prefix="train")
    scores = model.evaluate(preprocessing_fnc(data.x_dev), data.y_dev, verbose=2)
    experiment.log_metrics({"loss": scores[0],
                            "acc": scores[1]}, prefix="dev")
    timer = ElapsedTime("Test prediction")
    with timer:
        scores = model.evaluate(preprocessing_fnc(data.x_test), data.y_test, verbose=2)
    experiment.log_metric("test_inference_time", timer.elapsed_time_ms)
    experiment.log_metrics({"loss": scores[0],
                            "acc": scores[1]}, prefix="test")

import os
from abc import abstractmethod
from typing import Callable, List, Tuple

import numpy as np
from comet_ml import Experiment
from keras.models import Sequential
from keras.optimizers import Adam, SGD
from keras.callbacks import Callback, CSVLogger, EarlyStopping, LearningRateScheduler, LambdaCallback
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model

from mutil import ElapsedTime, KeepBest, LearningRateFinder
from mutil.DataSets import DataSets, get_cifar10_data

from models import get_model


class Trainer:

    def __init__(self, experiment) -> None:
        super().__init__()
        self.experiment = experiment

    def train(self, experiment: Experiment, epochs: int, batch_size: int):
        self.initial_epoch = 0
        self.training_datagen = self._create_training_data_generator()
        self.data = self._get_data()
        self.model = self._create_model()
        self.callbacks = self._create_callbacks(epochs, batch_size)

        self._pre_training(epochs, batch_size)

        self.model.fit_generator(
            self.training_datagen.flow(self.data.x_train, self.data.y_train, batch_size=batch_size),
            steps_per_epoch=len(self.data.x_train) / batch_size,
            epochs=epochs,
            validation_data=(self.training_datagen.preprocessing_function(self.data.x_dev), self.data.y_dev),
            shuffle=True,
            callbacks=self.callbacks,
            verbose=2,
            initial_epoch=self.initial_epoch)

        self._log_final_metrics(experiment, self.model, self.data, self.training_datagen.preprocessing_function)

        self._post_training(self.model)

    def _log_final_metrics(self, experiment: Experiment, model: Sequential, data: DataSets,
                           preprocessing_fnc: Callable[[np.ndarray], np.ndarray]):
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

    @abstractmethod
    def _create_training_data_generator(self) -> ImageDataGenerator:
        pass

    @abstractmethod
    def _get_data(self) -> DataSets:
        pass

    @abstractmethod
    def _create_model(self) -> Sequential:
        pass

    @abstractmethod
    def _create_callbacks(self, epochs: int, batch_size: int) -> List[Callback]:
        pass

    @abstractmethod
    def _pre_training(self, epochs: int, batch_size: int) -> None:
        pass

    @abstractmethod
    def _post_training(self, model) -> None:
        pass


class Cifar10Trainer(Trainer):
    def __init__(self, experiment: Experiment, learning_rate: float, data_portion: float = 1.0,
                 scheduler: Callable[[int], float] = None) -> None:
        super().__init__(experiment)
        self.learning_rate = learning_rate
        self.data_portion = data_portion
        self.scheduler = scheduler

        self.log_path, self.model_path = self.__get_output_paths()

    def __get_output_paths(self) -> Tuple[str, str]:
        save_dir = os.path.join(os.getcwd(), 'results')
        name = self.experiment.get_key()
        model_name = '{}.h5'.format(name)
        log_name = '{}.csv'.format(name)
        model_path = os.path.join(save_dir, model_name)
        log_path = os.path.join(save_dir, log_name)
        return log_path, model_path

    def _create_training_data_generator(self) -> ImageDataGenerator:
        preprocessing_fnc = lambda x: x.astype('float32') / 255.0
        return ImageDataGenerator(preprocessing_function=preprocessing_fnc)

    def _get_data(self) -> DataSets:
        return get_cifar10_data(data_portion=self.data_portion)

    def _create_model(self) -> Sequential:
        opt = SGD(lr=self.learning_rate)
        model = get_model()
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])

        self.log_model_plot(model)
        return model

    def log_model_plot(self, model):
        model_plot_file_name = 'model.png'
        plot_model(model, model_plot_file_name, show_shapes=True)
        self.experiment.log_image(model_plot_file_name)
        os.remove(model_plot_file_name)

    def _create_callbacks(self, epochs: int, batch_size: int) -> List[Callback]:
        csv_cb = CSVLogger(self.log_path)
        keep_best_cb = KeepBest('val_acc')
        early_stopping_cb = EarlyStopping('val_acc', patience=250, restore_best_weights=True, verbose=2)

        callbacks = [csv_cb, early_stopping_cb, keep_best_cb]
        if self.scheduler is not None:
            self._log_scheduler(epochs)
            callbacks.append(LearningRateScheduler(self.scheduler))

        return callbacks

    def _log_scheduler(self, epochs: int):
        """
        Compute the set of learning rates for each corresponding epochs and logs it in experiment.
        """
        for i in range(epochs):
            self.experiment.log_parameter('learning rate', self.scheduler(i), i)

    def _post_training(self, model) -> None:
        model.save(self.model_path)
        self.experiment.log_asset(self.model_path)
        self.experiment.log_asset(self.log_path)


class Cifar10FindLR(Cifar10Trainer):
    def __init__(self, experiment: Experiment, learning_rate: float, data_portion: float = 1.0,
                 scheduler: Callable[[int], float] = None, max_learning_rate: float = 10e-1) -> None:
        super().__init__(experiment, learning_rate, data_portion, scheduler)
        self.max_learning_rate = max_learning_rate

    def _create_callbacks(self, epochs: int, batch_size: int) -> List[Callback]:
        callbacks = super()._create_callbacks(epochs, batch_size)
        self.lrf = LearningRateFinder(model=self.model)
        self.lrf.lrMult = (self.max_learning_rate / self.learning_rate) ** (
                    1.0 / (epochs * len(self.data.x_train) / batch_size))
        callbacks.append(self.lrf)

        return callbacks

    def _post_training(self, model) -> None:
        super()._post_training(model)
        self.experiment.log_figure('lr vs acc', self.lrf.plot_loss())

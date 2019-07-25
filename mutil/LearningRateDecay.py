import holoviews as hv
import numpy as np
from abc import abstractmethod
from typing import List

from comet_ml import Experiment


class LearningRateDecay:
    def plot(self, epochs: List[int], title: str = "Learning Rate Schedule") -> hv.Layout:
        """
        Compute the set of learning rates for each corresponding epochs and plots it.
        :param epochs: Compute the set of learning rates for each corresponding epoch.
        :param title: The title of the plot.
        :return:
        """
        lrs = [self(i) for i in epochs]
        data = {'learning rate': lrs, 'epoch': epochs}
        return hv.Curve(data, kdims='epoch', vdims='learning rate').opts(title=title)

    def experiment_log(self, experiment: Experiment, epochs: List[int]):
        """
        Compute the set of learning rates for each corresponding epochs and logs it in experiment.
        :param experiment: CometML experiment
        :param epochs: Compute the set of learning rates for each corresponding epoch.
        """
        for i in epochs:
            experiment.log_parameter('learning rate', self(i), i)

    @abstractmethod
    def __call__(self, epoch: int) -> float:
        """
        Compute the learning rate for the current epoch
        :param epoch:
        :return:
        """
        pass


class StepDecay(LearningRateDecay):
    def __init__(self, init_alpha: float = 0.01, factor: float = 0.25, drop_every: int = 10):
        """
        Step based learning rate decay.
        :param init_alpha: Initial learning rate
        :param factor: Learning rate reduction factor
        :param drop_every: Number of epochs between reductions.
        """
        self.initAlpha = init_alpha
        self.factor = factor
        self.dropEvery = drop_every

    def __call__(self, epoch: int) -> float:
        """
        Compute the learning rate for the current epoch
        :param epoch:
        :return:
        """
        exp = np.floor((1 + epoch) / self.dropEvery)
        alpha = self.initAlpha * (self.factor ** exp)

        return float(alpha)


class PolynomialDecay(LearningRateDecay):
    def __init__(self, max_epochs: int, init_alpha: float = 0.01, power: float = 1.0):
        """
        Polynomial based learning rate decay.
        :param max_epochs: Number of epochs.
        :param init_alpha: Initial learning rate
        :param power: Power of the polynomial. (1.0 is linear)
        """
        self.maxEpochs = max_epochs
        self.initAlpha = init_alpha
        self.power = power

    def __call__(self, epoch: int) -> float:
        """
        Compute the learning rate for the current epoch
        :param epoch:
        :return:
        """
        decay = (1 - (epoch / float(self.maxEpochs))) ** self.power
        alpha = self.initAlpha * decay

        return float(alpha)

from typing import List

from comet_ml import Experiment

from .LearningRateDecay import LearningRateDecay


class CyclicLearningRate:
    def __init__(self, learning_rate_decay: LearningRateDecay, reset_epoch: int):
        """
        Cyclic learning rate algorithm. The learning rate is restarted the given times.
        :param learning_rate_decay: Learning rate decay algorithm
        :param reset_epoch: number of epoch between resets
        """
        self.reset_epoch = reset_epoch
        self.learning_rate_decay = learning_rate_decay

    def __call__(self, epoch: int) -> float:
        """
        Compute the learning rate for the current epoch
        :param epoch:
        :return:
        """
        return self.learning_rate_decay(epoch % self.reset_epoch)

    def experiment_log(self, experiment: Experiment, epochs: List[int]):
        """
        Compute the set of learning rates for each corresponding epochs and logs it in experiment.
        :param experiment: CometML experiment
        :param epochs: Compute the set of learning rates for each corresponding epoch.
        """
        for i in epochs:
            experiment.log_parameter('learning rate', self(i), i)


class TentCyclicLearningRate(CyclicLearningRate):
    def __call__(self, epoch: int) -> float:
        tent_break_epoch = int(self.reset_epoch / 2)
        if epoch % self.reset_epoch < tent_break_epoch:
            return self.learning_rate_decay(epoch % self.reset_epoch)
        else:
            return self.learning_rate_decay(tent_break_epoch - (epoch % tent_break_epoch))

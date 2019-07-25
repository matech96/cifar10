from .LearningRateDecay import LearningRateDecay


class CyclicLearningRate:
    def __init__(self, learning_rate_decay: LearningRateDecay, n_resets: int):
        """
        Cyclic learning rate algorithm. The learning rate is restarted the given times.
        :param learning_rate_decay: Learning rate decay algorithm
        :param n_resets: number of resets (Total epochs = learning_rate_decay.epochs * n_resets)
        """
        self.learning_rate_decay = learning_rate_decay
        self.n_resets = n_resets

    def __call__(self, epoch: int) -> float:
        """
        Compute the learning rate for the current epoch
        :param epoch:
        :return:
        """
        return self.learning_rate_decay(epoch % self.n_resets)

from comet_ml import Experiment

from mutil.CyclicLearningRate import TentCyclicLearningRate
from mutil.LearningRateDecay import PolynomialDecay
from mutil.Trainer import Cifar10Trainer, Cifar10FindLR

for tent_bent_epoch in [10, 20, 30, 40]:
    experiment = Experiment(api_key="cgss7piePhyFPXRw1J2uUEjkQ",
                            project_name="cifar10-25-auto-lr", workspace="matech96")
    experiment.set_name(f'CyclicLR_{tent_bent_epoch}')
    scheduler = TentCyclicLearningRate(PolynomialDecay(max_epochs=tent_bent_epoch, init_alpha=1e-1, min_alpha=1e-4),
                                       reset_epoch=tent_bent_epoch * 2)
    trainer = Cifar10Trainer(learning_rate=10e-10, experiment=experiment, scheduler=scheduler)
    trainer.train(experiment=experiment, epochs=3, batch_size=64)

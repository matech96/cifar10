from comet_ml import Experiment

from mutil.Trainer import Cifar10Trainer, Cifar10FindLR

experiment = Experiment(api_key="cgss7piePhyFPXRw1J2uUEjkQ",
                        project_name="cifar10-25-auto-lr", workspace="matech96")
experiment.set_name('testfinlr')
trainer = Cifar10FindLR(learning_rate=10e-10, max_learning_rate=10e2, experiment=experiment)
trainer.train(experiment=experiment, epochs=1, batch_size=64)

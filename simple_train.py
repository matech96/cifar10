from comet_ml import Experiment

from mutil.Trainer import Cifar10Trainer

experiment = Experiment(api_key="cgss7piePhyFPXRw1J2uUEjkQ",
                        project_name="cifar10-25-auto-lr", workspace="matech96")
experiment.set_name('test')
trainer = Cifar10Trainer(learning_rate=0.001)
trainer.train(experiment=experiment, epochs=1, batch_size=64)

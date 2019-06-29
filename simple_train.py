from comet_ml import Experiment

from train import train_cifar10

experiment = Experiment(api_key="cgss7piePhyFPXRw1J2uUEjkQ",
                        project_name="general", workspace="matech96")
experiment.set_name('long train for early stopping')

train_cifar10(batch_size=64, learning_rate=0.001, epochs=1500, experiment=experiment)

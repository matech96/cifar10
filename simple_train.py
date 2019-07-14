from comet_ml import Experiment
from keras import regularizers

from train import train_cifar10, get_model

experiment = Experiment(api_key="cgss7piePhyFPXRw1J2uUEjkQ",
                        project_name="cifar10-07-initialization", workspace="matech96")
experiment.set_name("he_normal")
model = get_model(dropout_rate=0.5)
train_cifar10(batch_size=64, learning_rate=0.001, epochs=10000, experiment=experiment, model=model)

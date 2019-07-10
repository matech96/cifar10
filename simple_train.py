from comet_ml import Experiment
from keras import regularizers

from train import train_cifar10, get_model

experiment = Experiment(api_key="cgss7piePhyFPXRw1J2uUEjkQ",
                        project_name="cifar10-02-l1-l2", workspace="matech96")
experiment.set_name("long")
model = get_model(regularization=regularizers.l1_l2(l1=0.0001, l2=0.01))
train_cifar10(batch_size=64, learning_rate=0.001, epochs=10000, experiment=experiment, model=model)

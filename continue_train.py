from comet_ml import ExistingExperiment
from keras import regularizers

from train import train_cifar10, get_model

experiment = ExistingExperiment(api_key="cgss7piePhyFPXRw1J2uUEjkQ",
                                previous_experiment="4af0f6389f1e4868aa8c9a189cf59472")
experiment.set_name("long")
model = get_model(regularization=regularizers.l1_l2(l1=0.0001, l2=0.01))
model.load_weights('results/4af0f6389f1e4868aa8c9a189cf59472.h5')
train_cifar10(batch_size=64, learning_rate=0.001, epochs=10000, experiment=experiment, model=model)

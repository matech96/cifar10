from comet_ml import Optimizer
from train import train_cifar10, get_model_5block
from keras import regularizers

config = {
    "algorithm": "grid",
    "name": "Dropout testing",

    "spec": {
        "metric": "acc",
    },

    "parameters": {
        "dropout_rate": {"type": "discrete", "values": [0, 0.25, 0.5, 0.75, 0.9]},
    },
}
optimizer = Optimizer(config, api_key="cgss7piePhyFPXRw1J2uUEjkQ", project_name="cifar10_dropout")

for experiment in optimizer.get_experiments():
    dropout_rate = experiment.get_parameter("dropout_rate")
    train_cifar10(batch_size=64, learning_rate=0.001, epochs=1000, experiment=experiment, dropout_rate=dropout_rate)

from comet_ml import Optimizer
from train import train_cifar10, get_model
from keras import regularizers

config = {
    "algorithm": "grid",
    "name": "Dropout testing",

    "spec": {
        "metric": "acc",
    },

    "parameters": {
        "dropout_rate": {"type": "discrete", "values": [0.55, 0.6, 0.65, 0.7]},
    },
}
optimizer = Optimizer(config, api_key="cgss7piePhyFPXRw1J2uUEjkQ", project_name="cifar10-06-dropout")

for experiment in optimizer.get_experiments():
    dropout_rate = experiment.get_parameter("dropout_rate")
    experiment.set_name(str(dropout_rate))
    model = get_model(dropout_rate=dropout_rate)
    train_cifar10(batch_size=64, learning_rate=0.001, epochs=1000, experiment=experiment, model=model)

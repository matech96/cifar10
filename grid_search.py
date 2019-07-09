from comet_ml import Optimizer
from train import train_cifar10, get_model_5block
from keras import regularizers

config = {
    "algorithm": "grid",
    "name": "5block batch size and learning rate",

    "spec": {
        "metric": "acc",
    },

    "parameters": {
        "batch_size": {"type": "discrete", "values": [8, 16, 32]},
        "learning_rate": {"type": "discrete", "values": [0.0001, 0.00001]},
    },
}
optimizer = Optimizer(config, api_key="cgss7piePhyFPXRw1J2uUEjkQ", project_name="cifar10_block5_bs_lr")

for experiment in optimizer.get_experiments():
    batch_size = experiment.get_parameter("batch_size")
    learning_rate = experiment.get_parameter("learning_rate")
    train_cifar10(batch_size=batch_size, learning_rate=learning_rate, epochs=1000, experiment=experiment, model_fnc=get_model_5block)

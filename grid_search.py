from comet_ml import Optimizer
from train import train_cifar10

config = {
    "algorithm": "grid",
    "name": "Batch size 1-1024, 10 epoch",

    "spec": {
        "metric": "acc",
    },

    "parameters": {
        "batch_size": {"type": "discrete", "values": [16, 32, 64, 128, 256, 512, 1024]},
        "learning_rate": {"type": "discrete", "values": [0.01, 0.001, 0.0001]},
    },
}
optimizer = Optimizer(config, api_key="cgss7piePhyFPXRw1J2uUEjkQ", project_name="cifar10_batch_size")

for experiment in optimizer.get_experiments():
    batch_size = experiment.get_parameter("batch_size")
    learning_rate = experiment.get_parameter("learning_rate")
    epochs = 10
    train_cifar10(batch_size=batch_size, learning_rate=learning_rate, epochs=epochs, experiment=experiment)

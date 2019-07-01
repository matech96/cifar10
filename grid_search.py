from comet_ml import Optimizer
from train import train_cifar10
from keras import regularizers

config = {
    "algorithm": "grid",
    "name": "L1 and L2 regularization",

    "spec": {
        "metric": "acc",
    },

    "parameters": {
        "l1": {"type": "discrete", "values": [1, 0.1, 0.01, 0.001, 0.0001]},
        "l2": {"type": "discrete", "values": [1, 0.1, 0.01, 0.001, 0.0001]},
    },
}
optimizer = Optimizer(config, api_key="cgss7piePhyFPXRw1J2uUEjkQ", project_name="cifar10_l1_l2")

for experiment in optimizer.get_experiments():
    l1 = experiment.get_parameter("l1")
    l2 = experiment.get_parameter("l2")
    epochs = 10
    train_cifar10(batch_size=64, learning_rate=0.001, epochs=1000, experiment=experiment,
                  regularization=regularizers.l1_l2(l1=l1, l2=l2))

from comet_ml import Optimizer
from train import train_cifar10, get_model
from keras import regularizers

config = {
    "algorithm": "grid",
    "name": "L1, L2 regularization without dropout",

    "spec": {
        "metric": "dev_acc",
    },

    "parameters": {
        "l1": {"type": "discrete", "values": [0]},
        "l2": {"type": "discrete", "values": [0.01, 0.001, 0.0001]},
    },
}
optimizer = Optimizer(config, api_key="cgss7piePhyFPXRw1J2uUEjkQ", project_name="cifar10-08-l1-l2-nodo")

for experiment in optimizer.get_experiments():
    l1 = experiment.get_parameter("l1")
    l2 = experiment.get_parameter("l2")
    experiment.set_name('{}_{}'.format(l1, l2))
    model = get_model(regularization=regularizers.l1_l2(l1=l1, l2=l2), dropout_rate=0)
    train_cifar10(batch_size=64, learning_rate=0.001, epochs=1000, experiment=experiment, model=model)

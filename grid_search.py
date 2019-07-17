from comet_ml import Optimizer
from train import train_cifar10
from models import get_model

config = {
    "algorithm": "grid",
    "name": "Batch size - power of 2",

    "spec": {
        "metric": "acc",
    },

    "parameters": {
        "batch_size": {"type": "discrete",
                       "values": list(range(34, 48)) + list(range(50, 64))},
    },
}
optimizer = Optimizer(config, api_key="cgss7piePhyFPXRw1J2uUEjkQ", project_name="cifar10-11-bs-power2")

for experiment in optimizer.get_experiments():
    batch_size = experiment.get_parameter("batch_size")
    experiment.set_name(batch_size)
    model = get_model()
    train_cifar10(batch_size=batch_size, learning_rate=0.001, epochs=10, experiment=experiment, model=model)

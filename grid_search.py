from comet_ml import Optimizer
from train import train_cifar10, get_model

config = {
    "algorithm": "grid",
    "name": "Dropout testing",

    "spec": {
        "metric": "acc",
    },

    "parameters": {
        "initialization": {"type": "categorical",
                           "values": ['TruncatedNormal', 'VarianceScaling', 'glorot_normal', 'glorot_uniform',
                                      'he_normal', 'he_uniform']},
    },
}
optimizer = Optimizer(config, api_key="cgss7piePhyFPXRw1J2uUEjkQ", project_name="cifar10-11-initialization")

for experiment in optimizer.get_experiments():
    initialization = experiment.get_parameter("initialization")
    experiment.set_name(initialization)
    model = get_model(dropout_rate=0.5, initialization=initialization)
    train_cifar10(batch_size=64, learning_rate=0.001, epochs=10000, experiment=experiment, model=model)

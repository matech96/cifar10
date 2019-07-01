from comet_ml import Experiment

from train import train_cifar10, \
    get_model, get_model_same, get_model_3block, get_model_4block, get_model_5block

experiment = Experiment(api_key="cgss7piePhyFPXRw1J2uUEjkQ",
                        project_name="architecture testing", workspace="matech96")

model_fncs = [get_model, get_model_same, get_model_3block, get_model_4block, get_model_5block]
experiment_names = ['base', 'same', '3block', '4block', '5block']

for name, model_fnc in zip(experiment_names, model_fncs):
    experiment.set_name(name)
    train_cifar10(batch_size=64, learning_rate=0.001, epochs=1000, experiment=experiment)

from comet_ml import ExistingExperiment
from keras.models import load_model

from train import train_cifar10

experiment_keys = ['74d6dcc313d145f6adcd4a71b75403e6', 'f348f3bfc52e4c8a8cd5093fa05c7fc9',
                   'e814a1cb2c3a4ff4bc23798f1d67e386', '0417b2dce5e44444869178b8d602d896']
for key in experiment_keys:
    experiment = ExistingExperiment(api_key="cgss7piePhyFPXRw1J2uUEjkQ",
                                    previous_experiment=key)
    model = load_model('results/{}.h5'.format(key))
    train_cifar10(batch_size=64, learning_rate=0.001, epochs=10000, experiment=experiment, model=model,
                  initial_epoch=1000)

from comet_ml import Optimizer
from keras.preprocessing.image import ImageDataGenerator

from train import train_cifar10
from models import get_model

config = {
    "algorithm": "grid",
    "name": "Batch size - power of 2",

    "spec": {
        "metric": "acc",
    },

    "parameters": {
        "fill_mode": {"type": "categorical",
                      "values": ['constant', 'nearest', 'reflect', 'wrap']},
    },
}
optimizer = Optimizer(config, api_key="cgss7piePhyFPXRw1J2uUEjkQ", project_name="cifar10-14-filling")

for experiment in optimizer.get_experiments():
    fill_mode = experiment.get_parameter("fill_mode")
    experiment.set_name(fill_mode)
    model = get_model()
    training_datagen = ImageDataGenerator(width_shift_range=0.15, fill_mode=fill_mode)
    train_cifar10(batch_size=64, learning_rate=0.001, epochs=10000, experiment=experiment, model=model,
                  training_datagen=training_datagen)

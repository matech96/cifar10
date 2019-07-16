from comet_ml import Optimizer
from keras.preprocessing.image import ImageDataGenerator

from train import train_cifar10
from models import get_model

config = {
    "algorithm": "grid",
    "name": "Datagen - horizontal shifting",

    "spec": {
        "metric": "dev_acc",
        "maxCombo": 5,
    },

    "parameters": {
        "width_shift_range": {"type": "float",
                              "min": 0,
                              "max": 0.5},
    },
}
optimizer = Optimizer(config, api_key="cgss7piePhyFPXRw1J2uUEjkQ", project_name="cifar10-13-horizontal-shifting")

for experiment in optimizer.get_experiments():
    width_shift_range = experiment.get_parameter("width_shift_range")
    experiment.set_name("{}".format(width_shift_range))
    model = get_model()
    training_datagen = ImageDataGenerator(width_shift_range=width_shift_range)
    train_cifar10(batch_size=64, learning_rate=0.001, epochs=1000, experiment=experiment, model=model,
                  training_datagen=training_datagen)

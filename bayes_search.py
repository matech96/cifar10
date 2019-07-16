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
    horizontal_flip = experiment.get_parameter("horizontal_flip")
    vertical_flip = experiment.get_parameter("vertical_flip")
    experiment.set_name("{}_{}".format(vertical_flip, horizontal_flip))
    model = get_model()
    training_datagen = ImageDataGenerator(horizontal_flip=horizontal_flip, vertical_flip=vertical_flip)
    train_cifar10(batch_size=64, learning_rate=0.001, epochs=1000, experiment=experiment, model=model,
                  training_datagen=training_datagen)

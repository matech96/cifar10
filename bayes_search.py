from comet_ml import Optimizer
from keras.preprocessing.image import ImageDataGenerator

from train import train_cifar10
from models import get_model

config = {
    "algorithm": "grid",
    "name": "Datagen - rotation",

    "spec": {
        "metric": "dev_acc",
        # "maxCombo": 5,
    },

    "parameters": {
        "rotation_range": {"type": "discrete",
                           "values": [0, 10, 20, 30]},
    },
}
optimizer = Optimizer(config, api_key="cgss7piePhyFPXRw1J2uUEjkQ", project_name="cifar10-15-rotation")

for experiment in optimizer.get_experiments():
    rotation_range = experiment.get_parameter("rotation_range")
    experiment.set_name("{}".format(rotation_range))
    model = get_model()
    training_datagen = ImageDataGenerator(rotation_range=rotation_range)
    train_cifar10(batch_size=64, learning_rate=0.001, epochs=10000, experiment=experiment, model=model,
                  training_datagen=training_datagen)

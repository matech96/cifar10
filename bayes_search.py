from comet_ml import Optimizer
from keras.preprocessing.image import ImageDataGenerator

from train import train_cifar10
from models import get_model

config = {
    "algorithm": "grid",
    "name": "Datagen - zoom",

    "spec": {
        "metric": "dev_acc",
        # "maxCombo": 5,
    },

    "parameters": {
        "zoom_range": {"type": "discrete",
                       "values": [0.3]},
    },
}
optimizer = Optimizer(config, api_key="cgss7piePhyFPXRw1J2uUEjkQ", project_name="cifar10-16-zoom")

for experiment in optimizer.get_experiments():
    zoom_range = experiment.get_parameter("zoom_range")
    experiment.set_name("{}".format(zoom_range))
    model = get_model()
    training_datagen = ImageDataGenerator(zoom_range=zoom_range)
    train_cifar10(batch_size=64, learning_rate=0.001, epochs=10000, experiment=experiment, model=model,
                  training_datagen=training_datagen)

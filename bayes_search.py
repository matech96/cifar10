from comet_ml import Optimizer
from keras.preprocessing.image import ImageDataGenerator

from train import train_cifar10
from models import get_model

config = {
    "algorithm": "grid",
    "name": "Datagen - whitening",

    "spec": {
        "metric": "dev_acc",
        # "maxCombo": 5,
    },

    "parameters": {
        "zca_whitening": {"type": "discrete",
                          "values": [0, 1]},
    },
}
optimizer = Optimizer(config, api_key="cgss7piePhyFPXRw1J2uUEjkQ", project_name="cifar10-17-whitening")

for experiment in optimizer.get_experiments():
    zca_whitening = experiment.get_parameter("zca_whitening")
    experiment.set_name("{}".format(zca_whitening))
    model = get_model()
    training_datagen = ImageDataGenerator(zca_whitening=zca_whitening)
    train_cifar10(batch_size=64, learning_rate=0.001, epochs=10000, experiment=experiment, model=model,
                  training_datagen=training_datagen)

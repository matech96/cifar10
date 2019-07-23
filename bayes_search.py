from comet_ml import Optimizer
from keras.preprocessing.image import ImageDataGenerator

from train import train_cifar10
from models import get_model

config = {
    "algorithm": "grid",
    "name": "Datagen - brightness",

    "spec": {
        "metric": "dev_acc",
        # "maxCombo": 5,
    },

    "parameters": {
        "brightness_range": {"type": "discrete",
                             "values": [0.0, 0.1, 0.2, 0.3]},
    },
}
optimizer = Optimizer(config, api_key="cgss7piePhyFPXRw1J2uUEjkQ", project_name="cifar10-20-brightness-repeat")

for experiment in optimizer.get_experiments():
    shear_range = experiment.get_parameter("shear_range")
    experiment.set_name("{}".format(shear_range))
    model = get_model()
    preprocessing_fnc = lambda x: x.astype('float32') / 255.0
    training_datagen = ImageDataGenerator(shear_range=shear_range,
                                          preprocessing_function=preprocessing_fnc)
    train_cifar10(batch_size=64, learning_rate=0.001, epochs=10000, experiment=experiment, model=model,
                  training_datagen=training_datagen)

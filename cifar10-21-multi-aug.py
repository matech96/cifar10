from comet_ml import Experiment
from keras.preprocessing.image import ImageDataGenerator

from train import train_cifar10
from models import get_model

parameters = {
    'horizontal_flip': True,
    'vertical_flip': False,
    'width_shift_range': 0.15,
    'height_shift_range': 0.15,
    'rotation_range': 0.0,
    'zoom_range': 0.0,
    'zca_whitening': False,
    'brightness_range': [0.8, 1.2],
    'shear_range': 0.0}

experiment = Experiment(api_key="cgss7piePhyFPXRw1J2uUEjkQ",
                        project_name="cifar10-21-multi-aug", workspace="matech96")
experiment.set_name("flip-shift-wh-rotate")
experiment.log_parameters(parameters)

model = get_model()
preprocessing_fnc = lambda x: x.astype('float32') / 255.0
training_datagen = ImageDataGenerator(preprocessing_function=preprocessing_fnc, **parameters)
train_cifar10(batch_size=64, learning_rate=0.001, epochs=10000, experiment=experiment, model=model,
              training_datagen=training_datagen)


from comet_ml import Experiment
from keras.preprocessing.image import ImageDataGenerator

from train import train_cifar10
from models import get_model

experiment = Experiment(api_key="cgss7piePhyFPXRw1J2uUEjkQ",
                        project_name="cifar10-17-whitening", workspace="matech96")
experiment.set_name("1")
zca_whitening = True
experiment.log_parameter("zca_whitening", zca_whitening)
model = get_model()
preprocessing_fnc = lambda x: x.astype('float32') / 255.0
training_datagen = ImageDataGenerator(zca_whitening=zca_whitening, preprocessing_function=preprocessing_fnc)
train_cifar10(batch_size=64, learning_rate=0.001, epochs=10000, experiment=experiment, model=model,
              training_datagen=training_datagen)

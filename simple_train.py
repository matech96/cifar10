from comet_ml import Experiment
from keras.preprocessing.image import ImageDataGenerator

from models import get_model
from mutil.CyclicLearningRate import TentCyclicLearningRate, CyclicLearningRate
from mutil.LearningRateDecay import PolynomialDecay
from train import train_cifar10

epochs = 3
experiment = Experiment(api_key="cgss7piePhyFPXRw1J2uUEjkQ",
                        project_name="cifar10-25-auto-lr", workspace="matech96")
experiment.set_name("base")
learning_rate = 0.001

model = get_model()
preprocessing_fnc = lambda x: x.astype('float32') / 255.0
training_datagen = ImageDataGenerator(preprocessing_function=preprocessing_fnc)
train_cifar10(batch_size=64, learning_rate=learning_rate, epochs=epochs, experiment=experiment,
              model=model, training_datagen=training_datagen, find_lr=True)

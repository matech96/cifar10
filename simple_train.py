from comet_ml import Experiment
from keras.preprocessing.image import ImageDataGenerator

from models import get_model
from mutil.LearningRateDecay import PolynomialDecay
from train import train_cifar10

experiment = Experiment(api_key="cgss7piePhyFPXRw1J2uUEjkQ",
                        project_name="cifar10-21-multi-aug", workspace="matech96")
experiment.set_name("linear")
epochs = 100
learning_rate = 0.001

model = get_model()
preprocessing_fnc = lambda x: x.astype('float32') / 255.0
training_datagen = ImageDataGenerator(preprocessing_function=preprocessing_fnc)
scheduler = PolynomialDecay(max_epochs=epochs, init_alpha=learning_rate, power=1.0)
train_cifar10(batch_size=64, learning_rate=learning_rate, epochs=epochs, experiment=experiment, model=model,
              training_datagen=training_datagen)

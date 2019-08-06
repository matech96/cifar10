from comet_ml import Experiment
from keras.preprocessing.image import ImageDataGenerator

from models import get_model
from mutil.CyclicLearningRate import TentCyclicLearningRate, CyclicLearningRate
from mutil.LearningRateDecay import PolynomialDecay
from train import train_cifar10

for data_portion in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
    epochs = 10000
    experiment = Experiment(api_key="cgss7piePhyFPXRw1J2uUEjkQ",
                            project_name="cifar10-24-data_set_size", workspace="matech96")
    experiment.set_name("{}".format(data_portion))
    experiment.log_parameter('data_portion', data_portion)
    learning_rate = 0.001

    model = get_model()
    preprocessing_fnc = lambda x: x.astype('float32') / 255.0
    training_datagen = ImageDataGenerator(preprocessing_function=preprocessing_fnc)
    train_cifar10(batch_size=64, learning_rate=learning_rate, epochs=epochs, experiment=experiment,
                  model=model, training_datagen=training_datagen, data_portion=data_portion)

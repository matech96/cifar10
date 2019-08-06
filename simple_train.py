from comet_ml import Experiment
from keras.preprocessing.image import ImageDataGenerator

from models import get_model
from mutil.CyclicLearningRate import TentCyclicLearningRate, CyclicLearningRate
from mutil.LearningRateDecay import PolynomialDecay
from train import train_cifar10

epochs = 300
experiment = Experiment(api_key="cgss7piePhyFPXRw1J2uUEjkQ",
                        project_name="cifar10-21-multi-aug", workspace="matech96")
experiment.set_name("cyclic_linear_flip_shift{}".format(epochs))
learning_rate = 0.001
n_resets = 4

model = get_model()
preprocessing_fnc = lambda x: x.astype('float32') / 255.0
training_datagen = ImageDataGenerator(preprocessing_function=preprocessing_fnc, horizontal_flip=True,
                                      width_shift_range=0.15)
scheduler = CyclicLearningRate(PolynomialDecay(max_epochs=epochs, init_alpha=learning_rate, power=1.0),
                               reset_epoch=epochs)
train_cifar10(batch_size=64, learning_rate=learning_rate, epochs=epochs * n_resets, experiment=experiment,
              model=model,
              training_datagen=training_datagen, scheduler=scheduler, early_stopping_th=None)

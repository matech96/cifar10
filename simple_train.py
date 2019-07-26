from comet_ml import Experiment
from keras.preprocessing.image import ImageDataGenerator

from models import get_model
from mutil.LearningRateDecay import PolynomialDecay
from mutil import CyclicLearningRate
from train import train_cifar10

for epochs in [100, 200, 300, 400]:
    experiment = Experiment(api_key="cgss7piePhyFPXRw1J2uUEjkQ",
                            project_name="cifar10-23-lr-schedule", workspace="matech96")
    experiment.set_name("cyclic_linear{}".format(epochs))
    learning_rate = 0.001

    model = get_model()
    preprocessing_fnc = lambda x: x.astype('float32') / 255.0
    training_datagen = ImageDataGenerator(preprocessing_function=preprocessing_fnc)
    scheduler = CyclicLearningRate(PolynomialDecay(max_epochs=epochs, init_alpha=learning_rate, power=1.0), n_resets=4)
    train_cifar10(batch_size=64, learning_rate=learning_rate, epochs=epochs, experiment=experiment, model=model,
                  training_datagen=training_datagen, scheduler=scheduler)

from comet_ml import Experiment
from comet_ml import Optimizer
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import CSVLogger
from keras.optimizers import Adam
import os

# experiment = Experiment(api_key="cgss7piePhyFPXRw1J2uUEjkQ",
#                         project_name="general", workspace="matech96")
# experiment.set_name('csl logger test long')

config = {
    "algorithm": "grid",
    "name": "Batch size 1-1024, 10 epoch",

    "spec": {
        "metric": "acc",
    },

    "parameters": {
        "batch_size": {"type": "discrete", "values": [16, 32, 64, 128, 256, 512, 1024]},
        "learning_rate": {"type": "discrete", "values": [0.01, 0.001, 0.0001]},
    },
}
opt = Optimizer(config, api_key="cgss7piePhyFPXRw1J2uUEjkQ", project_name="cifar10_batch_size")
for experiment in opt.get_experiments():
    batch_size = experiment.get_parameter("batch_size")
    learning_rate = experiment.get_parameter("learning_rate")
    num_classes = 10
    epochs = 10
    save_dir = os.path.join(os.getcwd(), 'results')
    model_name = '{}.h5'.format(experiment.get_key())
    log_name = '{}.csv'.format(experiment.get_key())
    model_path = os.path.join(save_dir, model_name)
    log_path = os.path.join(save_dir, log_name)

    # The data, split between train and test sets:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    opt = Adam(lr=learning_rate)

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    csv_cb = CSVLogger(log_path)
    callbacks = [csv_cb]

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True,
              callbacks=callbacks,
              verbose=2)

    model.save(model_path)
    experiment.log_asset(model_path)

    # Score trained model.
    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

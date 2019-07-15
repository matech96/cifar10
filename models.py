from keras import regularizers, Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Dropout, Flatten, Dense


def get_model(regularization: regularizers.Regularizer = None, dropout_rate: float = 0.5,
              initialization: str = 'VarianceScaling'):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=(32, 32, 3), kernel_initializer=initialization))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), kernel_regularizer=regularization, kernel_initializer=initialization))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_rate))
    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularization, kernel_initializer=initialization))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), kernel_regularizer=regularization, kernel_initializer=initialization))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_rate))
    model.add(Flatten())
    model.add(Dense(512, kernel_regularizer=regularization, kernel_initializer=initialization))
    model.add(Activation('relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(10, kernel_regularizer=regularization, kernel_initializer=initialization))
    model.add(Activation('softmax'))
    return model


def get_model_same(regularization: regularizers.Regularizer = None, dropout_rate: float = 0.5,
                   initialization: str = 'VarianceScaling'):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=(32, 32, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularization, kernel_initializer=initialization))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_rate))
    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularization, kernel_initializer=initialization))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularization, kernel_initializer=initialization))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_rate))
    model.add(Flatten())
    model.add(Dense(512, kernel_regularizer=regularization, kernel_initializer=initialization))
    model.add(Activation('relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(10, kernel_regularizer=regularization, kernel_initializer=initialization))
    model.add(Activation('softmax'))
    return model


def get_model_3block(regularization: regularizers.Regularizer = None, dropout_rate: float = 0.5,
                     initialization: str = 'VarianceScaling'):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=(32, 32, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_rate))

    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularization, kernel_initializer=initialization))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_rate))

    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularization, kernel_initializer=initialization))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_rate))

    model.add(Flatten())
    model.add(Dense(512, kernel_regularizer=regularization, kernel_initializer=initialization))
    model.add(Activation('relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(10, kernel_regularizer=regularization, kernel_initializer=initialization))
    model.add(Activation('softmax'))
    return model


def get_model_4block(regularization: regularizers.Regularizer = None, dropout_rate: float = 0.5,
                     initialization: str = 'VarianceScaling'):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=(32, 32, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_rate))

    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularization, kernel_initializer=initialization))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_rate))

    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularization, kernel_initializer=initialization))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_rate))

    model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularization, kernel_initializer=initialization))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_rate))

    model.add(Flatten())
    model.add(Dense(512, kernel_regularizer=regularization, kernel_initializer=initialization))
    model.add(Activation('relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(10, kernel_regularizer=regularization, kernel_initializer=initialization))
    model.add(Activation('softmax'))
    return model


def get_model_5block(regularization: regularizers.Regularizer = None, dropout_rate: float = 0.5,
                     initialization: str = 'VarianceScaling'):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=(32, 32, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_rate))

    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularization, kernel_initializer=initialization))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_rate))

    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularization, kernel_initializer=initialization))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_rate))

    model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularization, kernel_initializer=initialization))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_rate))

    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularization, kernel_initializer=initialization))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_rate))

    model.add(Flatten())
    model.add(Dense(512, kernel_regularizer=regularization, kernel_initializer=initialization))
    model.add(Activation('relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(10, kernel_regularizer=regularization, kernel_initializer=initialization))
    model.add(Activation('softmax'))
    return model

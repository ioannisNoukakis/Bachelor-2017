import keras
from keras.models import Sequential
from keras.layers import *
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K, optimizers

from img_loader import DatasetLoader
from model_utils import train_model


def create_n_run_mnist(dl: DatasetLoader, epochs=5):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=(28, 28, 3)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, activation='relu', border_mode="same", name="CAM"))
    model.add(GlobalAveragePooling2D(name="GAP"))
    model.add(Dense(dl.nb_classes, activation='softmax', name='W'))

    sgd = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    print(train_model(model, dl, epochs, batch_size=128))
    model.save("mnist.h5")

from keras.models import Sequential
from keras.layers import *
from pathlib import Path
from img_loader import *
import time

from logger import info
from model_utils import train_model


def get_custom_model(dataset_loader: DatasetLoader, mode, N_EPOCHS=5, random=False, save=None):
    """
    Creates and compile the custom model.

    :param dataset_loader: The data set loader with the model will train.
    :param mode: GAP for global average pooling layer or dense of the fully connected layers
    :param N_EPOCHS: the number of iterations over the data.
    :param random: if true the model will be returned with random weights.
    :param save: if not none the model will try first to load it's saved weights.
                 if such file does not exists, trains and creates it.
    :return: the model and its score
    """
    start = time.strftime("%c")
    the_true_score = []
    nb_classes = dataset_loader.get_nb_classes()

    # Define model architecture
    model = Sequential()

    model.add(Convolution2D(nb_filter=64, nb_row=3, nb_col=3, activation='relu', input_shape=(256, 256, 3),
                            dim_ordering='tf', name="Conv1"))
    model.add(Convolution2D(nb_filter=32, nb_row=3, nb_col=3, activation='relu', name="Conv2"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(nb_filter=32, nb_row=3, nb_col=3, activation='relu', name="Conv3"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(nb_filter=32, nb_row=3, nb_col=3, activation='relu', name="Conv4"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    if mode == "GAP":
        model.add(GlobalAveragePooling2D(name="GAP"))
    elif mode == "dense":
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax', name='W'))

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    if save is not None:
        w_file = Path(save + ".h5")
    else:
        w_file = Path("")
    model.summary()

    if random:
        return model

    if w_file.is_file():
        model.load_weights(save + ".h5")
        return model
    else:
        model, the_true_score = train_model(model, dataset_loader, N_EPOCHS, None)

        info("[CUSTOM MODEL] Model:", "")
        model.summary()
        info("[CUSTOM MODEL] Obtained the score:", the_true_score)
        info("[CUSTOM MODEL] Training started at:", start)
        info("[CUSTOM MODEL] Training ended at:", time.strftime("%c"))
        info("[CUSTOM MODEL] Classes:", nb_classes)
        info("[CUSTOM MODEL] Nb_epoch:", 10)
        if save is not None:
            model.save_weights("./"+save + ".h5")
        return model

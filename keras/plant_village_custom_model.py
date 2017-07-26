from keras import optimizers
from keras.models import Sequential
from keras.layers import *
from img_loader import *
import time
from model_utils import train_model


def train_custom_model(dataset_loader: DatasetLoader, mode="dense", N_EPOCHS=5):
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

    print("[CUSTOM MODEL]", mode, "selected.")

    if mode == "GAP":
        model.add(GlobalAveragePooling2D(name="GAP"))
    elif mode == "dense":
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax', name='W'))

    # Compile model
    sgd = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.summary()

    model, the_true_score = train_model(model, dataset_loader, N_EPOCHS)

    print("[CUSTOM MODEL] Model:", "")
    model.summary()
    print("[CUSTOM MODEL] Obtained the score:", the_true_score)
    print("[CUSTOM MODEL] Training started at:", start)
    print("[CUSTOM MODEL] Training ended at:", time.strftime("%c"))
    print("[CUSTOM MODEL] Classes:", nb_classes)
    print("[CUSTOM MODEL] Nb_epoch:", 10)
    model.save("custom_model.h5")

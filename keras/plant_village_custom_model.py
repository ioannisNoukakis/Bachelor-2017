from keras.models import Sequential
from keras.layers import *
from keras.utils import np_utils
from quiver_engine import server
from pathlib import Path
from img_loader import *
import time


def get_custom_model(img_u: DatasetLoader, mode, N_EPOCHS=5, random=False, save=None):
    start = time.strftime("%c")
    the_true_score = []
    nb_classes = img_u.get_nb_classes()

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
        model, the_true_score = train_model(model, img_u, N_EPOCHS, None)

        print("Model:")
        model.summary()
        print("Obtained the score:", the_true_score)
        print("Training started at:", start)
        print("Training ended at:", time.strftime("%c"))
        print("Classes:", nb_classes)
        print("Nb_epoch:", 10)
        if save is not None:
            model.save_weights("./"+save + ".h5")
        return model


def data_generator(img_u: DatasetLoader):
    _, x_train, y_train = img_u.load_dataset(no_test_data=True)
    # Preprocessing
    x_train = x_train.astype('float32')
    x_train /= 255

    y_train = np_utils.to_categorical(y_train, img_u.nb_classes)
    yield x_train, y_train


def train_model(model, img_u, n_epochs, callbacks):
    redo = True
    score = []

    print("Starting...")
    if callbacks:
        model.fit_generator(data_generator(img_u=img_u), img_u.number_of_imgs/10, epochs=n_epochs,
                            callbacks=callbacks, verbose=1)
        # model.fit(x_train, y_train, batch_size=10, nb_epoch=1, verbose=0, callbacks=callbacks)
    else:
        model.fit_generator(data_generator(img_u=img_u), img_u.number_of_imgs / 10, epochs=n_epochs, verbose=1)
        # model.fit(x_train, y_train, batch_size=10, nb_epoch=1, verbose=1)
    score = evaluate_model(model, img_u, score)
    redo = True
    return model, score


def evaluate_model(model, img_u, the_true_score):
    redo = True
    while redo:
        redo, x_test, y_test = img_u.load_dataset()
        # Preprocessing
        x_test = x_test.astype('float32')
        x_test /= 255

        y_test_2 = np_utils.to_categorical(y_test, img_u.nb_classes)
        # Evaluate model on test data
        the_true_score.append(model.evaluate(x_test, y_test_2, batch_size=10, verbose=1))
    return the_true_score

from keras.models import Sequential
from keras.layers import *
from keras.utils import np_utils
from quiver_engine import server
from pathlib import Path
from imgUtils import *
import time


def get_custom_model(mode):
    img_u = DatasetLoader("./dataset_rand", 10000)
    start = time.strftime("%c")
    the_true_score = []
    nb_classes = img_u.get_nb_classes()
    N_EPOCHS = 5

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

    if mode == "GAP":
        w_file = Path("./model_GAP.h5")
    elif mode == "dense":
        w_file = Path("./model_dense.h5")
    model.summary()

    if w_file.is_file() and mode == "GAP":
        model.load_weights("./model_GAP.h5")
        return model
    elif w_file.is_file() and mode == "dense":
        model.load_weights("./model_dense.h5")
        return model
    else:
        # Train
        redo = True
        # for i in range(0, N_EPOCHS):
        while redo:
            redo, x_train, y_train = img_u.load_dataset()
            # Preprocessing
            x_train = x_train.astype('float32')
            x_train /= 255

            y_train = np_utils.to_categorical(y_train, nb_classes)

            # Fit model on training data
            print("Starting...")
            model.fit(x_train, y_train, batch_size=10, nb_epoch=N_EPOCHS, verbose=1)
            # TODO: Maybe this is the wrong order of how to apply epochs -> investigate
            # redo = True

        print("Train completed! Will now evalutate...")

        # Evaluate
        redo = True
        while redo:
            redo, x_test, y_test = img_u.load_dataset()
            # Preprocessing
            x_test = x_test.astype('float32')
            x_test /= 255

            y_test_2 = np_utils.to_categorical(y_test, nb_classes)
            # Evaluate model on test data
            the_true_score.append(model.evaluate(x_test, y_test_2, batch_size=10, verbose=1))

            # y_hat = model.predict_classes(x_test)
            # print(pd.crosstab(y_hat, y_test))

        print("Model:")
        model.summary()
        print("Obtained the score:", the_true_score)
        print("Training started at:", start)
        print("Training ended at:", time.strftime("%c"))
        print("Classes:", nb_classes)
        print("Nb_epoch:", 10)
        model.save_weights("./model_GAP.h5")
        return model
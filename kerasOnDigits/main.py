from keras.models import Sequential
from keras.layers import *
from keras.utils import np_utils
from imgUtils import *
from mailsUtils import send_mail
import time
import sys

# https://elitedatascience.com/keras-tutorial-deep-learning-in-python#step-1


def main():

    np.random.seed(123)  # for reproducibility
    imgU = ImgUtils("./dataset")
    start = time.strftime("%c")
    theTrueScore = []

    for layout in range(0, 4):
        nb_classes, x_train, y_train, x_test, y_test = imgU.load_and_shuffle_dataset(layout)
        print(x_train.shape)
        # Preprocessing
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255

        y_train = np_utils.to_categorical(y_train, nb_classes)
        y_test = np_utils.to_categorical(y_test, nb_classes)

        print(x_train.shape)

        # Define model architecture
        model = Sequential()

        model.add(Convolution2D(nb_filter=64, nb_row=3, nb_col=3, activation='relu', input_shape=(1, 256, 256),
                                dim_ordering='th'))
        model.add(Convolution2D(nb_filter=64, nb_row=3, nb_col=3, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu'))
        model.add(Convolution2D(64, 3, 3, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(nb_classes, activation='softmax'))

        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Fit model on training data
        model.fit(x_train, y_train, batch_size=10, nb_epoch=1, verbose=0)

        # Evaluate model on test data
        theTrueScore.append(model.evaluate(x_test, y_test, verbose=0))
        print(model.evaluate(x_test, y_test, verbose=0))

    with open('log.txt', 'w') as f:
        sys.stdout = f
        print("Classes:", nb_classes)
        print("Model:")
        model.summary()
        print("Obtained the score:", theTrueScore)
        print("Training started at:", start)
        print("Training ended at:", time.strftime("%c"))
        sys.stdout = sys.__stdout__

    # with open('log.txt', 'r') as f:
        # send_mail("ioannisbachelorbot@gmail.com", "inoukakis@gmail.com", "<mdp>", f.read())
        # send_mail("ioannisbachelorbot@gmail.com", "inoukakis@gmail.com", "<mdp>", f.read())

if __name__ == "__main__":
    main()

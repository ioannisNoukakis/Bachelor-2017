from keras.models import Sequential
from keras.layers import *
from keras.utils import np_utils
from imgUtils import *

# https://elitedatascience.com/keras-tutorial-deep-learning-in-python#step-1


def reshaper(the_array):
    the_arrays = np.split(the_array, 4073)
    the_return_array = np.array([])
    for a in the_arrays:
        print(a.shape)
        a.T = a.T.reshape(a.shape[0], 1, 256, 256)
        the_return_array = np.concatenate((the_return_array, a), axis=0)

    return the_return_array


def main():

    np.random.seed(123)  # for reproducibility
    imgU = ImgUtils("./dataset")
    nb_classes, (X_train, y_train), (X_test, y_test) = imgU.load_and_shuffle_dataset()
    print(X_train.shape)

    # Preprocessing
    # X_train.T = reshaper(X_train)
    X_train = X_train.reshape(X_train.shape[0], 1, 256, 256)
    X_test = X_test.reshape(X_test.shape[0], 1, 256, 256)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    # Define model architecture
    model = Sequential()

    model.add(Convolution2D(nb_filter=64, nb_row=3, nb_col=3, activation='relu', input_shape=(1, 256, 256), dim_ordering='th'))
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
    model.fit(X_train, Y_train, batch_size=10, nb_epoch=10, verbose=1)

    # Evaluate model on test data
    score = model.evaluate(X_test, Y_test, verbose=0)
    print(score)

if __name__ == "__main__":
    main()

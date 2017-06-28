import random

import cv2
import keras
import numpy as np
from keras import applications
from keras.layers import GlobalAveragePooling2D, Dense
from keras.models import Sequential
import os
import math

from keras.utils import np_utils


class SimpleCallback(keras.callbacks.Callback):

    def on_batch_end(self, batch, logs={}):
        print("Batch ended!")


class ImagePath:
    """
    Container for image name and path and class.
    """
    def __init__(self, name, directory, img_class):
        self.name = name
        self.directory = directory
        self.img_class = img_class

    def get_name(self):
        return self.name

    def get_img_class(self):
        return self.img_class

    def get_directory(self):
        return self.directory


class DatasetLoader:
    """ Utility image loading class """

    def __init__(self, base_directory, max_img_loaded, no_training_data=False):
        """
        This utility expects you to have the following folder pattern:

        base_directory
        |__Class1
            |__Data1_class1
            |__Data2_class1
        |__Class2

        :param base_directory: The base directory
        :param max_img_loaded: The number of maximum images the utility can load at the same time..
        """
        self.baseDirectory = base_directory
        self.max_img_loaded = max_img_loaded
        self.no_training_data = no_training_data

        self.imgDataArray = []  # Array containing the path to the images to be loaded.
        self.number_of_imgs = 0
        self.number_of_imgs_for_train = 0
        self.number_of_imgs_for_test = 0
        self.iterator_path = 0  # used if you want to load one image at the time.
        self.i = 0  # iterator used when normal loading

        self.train_loaded = False

        print("[DATASET LOADER]", "Discovering dataset...")
        directories = next(os.walk(self.baseDirectory))[1]

        i = 0
        for directory in directories:
            for file_name in next(os.walk(self.baseDirectory + "/" + directory))[2]:
                self.imgDataArray.append(ImagePath(file_name, directory, i))
                self.number_of_imgs += 1
            i += 1

        print("[DATASET LOADER]", len(directories), "classes found.\n", self.number_of_imgs, "images found.")

        print("[DATASET LOADER]", "Shuffling order...")
        random.shuffle(self.imgDataArray)

        self.number_of_imgs_for_train = math.floor((self.number_of_imgs / 4) * 3)
        self.number_of_imgs_for_test = math.floor(self.number_of_imgs / 4)

        print("[DATASET LOADER]", "Ready for loading!\n", self.number_of_imgs_for_train, "for training and", self.number_of_imgs_for_test,
              "for testing")
        self.nb_classes = len(directories)

    def load_dataset(self):
        """
        Tries to load the dataset. If the dataset is too big for the memory split it.
        Call back this function until it does not return True as first return variable.
        :return: redo, score
        """

        data_x = []
        data_y = []
        redo = False
        j = 0
        while True:
            if not self.train_loaded and self.i == self.number_of_imgs_for_train:
                if self.no_training_data:
                    print("DATASET LOADER]", "Loaded all imgs for training.")
                    self.train_loaded = False
                    self.i = 0
                    redo = False
                    break
                else:
                    print("DATASET LOADER]", "Loaded all imgs for training. Next call will load test data...")
                    redo = False
                    self.train_loaded = True
                    break
            if self.i == self.number_of_imgs:
                print("DATASET LOADER]", "Loaded all imgs for test. Done! Next call will load train data")
                redo = False
                self.train_loaded = False
                self.i = 0
                break
            if j + 1 >= self.max_img_loaded:
                print("DATASET LOADER]", "")
                print("Max img loaded!", self.i, "/", self.number_of_imgs)
                redo = True
                break

            img = cv2.imread(self.baseDirectory + "/" + self.imgDataArray[self.i].get_directory() + "/" +
                             self.imgDataArray[self.i].get_name(), cv2.IMREAD_COLOR)
            data_x.append(img)
            data_y.append(self.imgDataArray[self.i].get_img_class())
            j += 1
            self.i += 1

        print("DATASET LOADER]", "Loading completed!")

        return redo, np.asarray(data_x), np.asarray(data_y)


def train_model(model, dataset_loader: DatasetLoader, n_epochs, callbacks):
    """
    Trains a model. At the end of each epochs evaluates it.
    :param model: The model to be trained
    :param dataset_loader: The data set loader with the model will train.
    :param n_epochs: The number of iterations over the data.
    :param callbacks: keras callbacks
    :return: The trained model and its score
    """
    score = []
    _, x_train, y_train = dataset_loader.load_dataset()
    # Preprocessing
    x_train = x_train.astype('float32')
    x_train /= 255

    y_train = np_utils.to_categorical(y_train, dataset_loader.nb_classes)

    # Fit model on training data
    print("[MODEL-UTILS] Starting...", "")
    if callbacks:
        model.fit(x_train, y_train, batch_size=10, epochs=n_epochs, verbose=1, callbacks=callbacks)
    else:
        model.fit(x_train, y_train, batch_size=10, nb_epoch=n_epochs, verbose=1)

    score = evaluate_model(model, dataset_loader, score)
    return model, score


def evaluate_model(model, dataset_loader: DatasetLoader, score):
    """
    Evaluates a model.

    :param model: The model to be evaluated.
    :param dataset_loader: The data set loader that will provide the evaluation data.
    :param score: The model's score
    :return: the new score
    """
    redo = True
    while redo:
        redo, x_test, y_test = dataset_loader.load_dataset()
        # Preprocessing
        x_test = x_test.astype('float32')
        x_test /= 255

        y_test_2 = np_utils.to_categorical(y_test, dataset_loader.nb_classes)
        # Evaluate model on test data
        score.append(model.evaluate(x_test, y_test_2, batch_size=10, verbose=1))
    return score


def main():
    np.random.seed(123)  # for reproducibility
    random.seed(123)

    data_loader = DatasetLoader("./dataset_mini", 10000)

    model = Sequential(applications.VGG16(weights='imagenet', include_top=False).layers)
    model.add(GlobalAveragePooling2D(name="GAP"))
    model.add(Dense(data_loader.nb_classes, activation='softmax', name='W'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    callback = SimpleCallback()

    train_model(model, data_loader, 10, [callback])

if __name__ == "__main__":
    main()
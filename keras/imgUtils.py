import os
import random
import cv2
import numpy as np
import math


def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


class ImagePath:
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

    def __init__(self, base_directory, max_img_loaded):
        """
        This utility expects you to have the following folder pattern:

        base_directory
        |__Class1
            |__Data1_class1
            |__Data2_class1
        |__Class2

        :param base_directory: The base directory
        :param max_img_loaded: The number of maximum images the utility can load at the same time.
        """
        self.baseDirectory = base_directory
        self.max_img_loaded = max_img_loaded

        self.number_of_image_read = 0
        self.imgDataArray = []  # Array containing the path to the images to be loaded.
        self.number_of_imgs = 0
        self.number_of_imgs_for_train = 0
        self.number_of_imgs_for_test = 0
        self.iterator_path = 0  # used if you want to load one image at the time.

        self.train_loaded = False

        print("Discovering dataset...")
        directories = next(os.walk(self.baseDirectory))[1]

        i = 0
        for directory in directories:
            for file_name in next(os.walk(self.baseDirectory + "/" + directory))[2]:
                self.imgDataArray.append(ImagePath(file_name, directory, i))
                self.number_of_imgs += 1
            i += 1

        print(len(directories), "classes found.\n", self.number_of_imgs, "images found.")

        print("Shuffling order...")
        random.shuffle(self.imgDataArray)

        self.number_of_imgs_for_train = math.floor((self.number_of_imgs / 4) * 3)
        self.number_of_imgs_for_test = math.floor(self.number_of_imgs / 4)

        print("Ready for loading!\n", self.number_of_imgs_for_train, "for training and", self.number_of_imgs_for_test,
              "for testing")
        self.nb_classes = len(directories)

    def get_nb_classes(self):
        return self.nb_classes

    def has_next(self):
        return self.iterator_path + 1 < self.number_of_imgs

    def next_path(self):
        """Return the path one image at each call in the order defined in the constructor"""
        p = self.imgDataArray[self.iterator_path]
        self.iterator_path += 1
        return self.baseDirectory + "/" + p.get_directory() + "/" + p.get_name()

    def load_dataset(self):

        data_x = []
        data_y = []
        redo = False
        j = 0
        for img_data in self.imgDataArray:
            if not self.train_loaded and self.number_of_image_read == self.number_of_imgs_for_train:
                print("Loaded all imgs for training. Next call will load test data...")
                redo = False
                self.train_loaded = True
                break
            if self.number_of_image_read == self.number_of_imgs:
                print("Loaded all imgs for test. Done! Next call will load train data")
                redo = False
                self.train_loaded = False
                break
            if j + 1 >= self.max_img_loaded:
                print("Max img loaded!", self.number_of_image_read, "/", self.number_of_imgs)
                redo = True
                break

            img = cv2.imread(self.baseDirectory + "/" + img_data.get_directory() + "/" + img_data.get_name(),
                             cv2.IMREAD_COLOR)
            data_x.append(img)
            data_y.append(img_data.get_img_class())
            j += 1
            self.number_of_image_read += 1

        print("Loading completed!")

        return redo, np.asarray(data_x), np.asarray(data_y)

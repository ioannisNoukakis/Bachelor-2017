import os
import random
import cv2
import numpy as np
import math
from logger import info


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

        Sometimes the data is too big to fit in memory. So this utility will split it
        and you'ill have to call load_dataset as longs as it returns True as it first
        return var.

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

        info("DATASET LOADER]", "Discovering dataset...")
        directories = next(os.walk(self.baseDirectory))[1]

        i = 0
        for directory in directories:
            for file_name in next(os.walk(self.baseDirectory + "/" + directory))[2]:
                self.imgDataArray.append(ImagePath(file_name, directory, i))
                self.number_of_imgs += 1
            i += 1

        info("DATASET LOADER]", "")
        print(len(directories), "classes found.\n", self.number_of_imgs, "images found.")

        info("DATASET LOADER]", "Shuffling order...")
        random.shuffle(self.imgDataArray)

        self.number_of_imgs_for_train = math.floor((self.number_of_imgs / 4) * 3)
        self.number_of_imgs_for_test = math.floor(self.number_of_imgs / 4)

        info("DATASET LOADER]", "")
        print("Ready for loading!\n", self.number_of_imgs_for_train, "for training and", self.number_of_imgs_for_test,
              "for testing")
        self.nb_classes = len(directories)

    def get_nb_classes(self):
        """
        Returns the number of classes in the dataset.
        :return: the number of classes.
        """
        return self.nb_classes

    def get_nb_images(self):
        """
        Returns the number of images in the dataset.
        :return: the number of images.
        """
        return self.number_of_imgs

    def has_next(self):
        """
        Check for the iterator path if it has a next.
        :return: True if the iterator has a next
        """
        return self.iterator_path + 1 < self.number_of_imgs

    def has_next_in_order(self):
        """
        Checks for the order iterator path if it has a next.
        :return:
        """
        return self.i + 1 < self.number_of_imgs

    def get(self, index: int):
        """
        Return an image path at the given index.
        :param index: the index
        :return: and image path
        """
        p = self.imgDataArray[index]
        return self.baseDirectory + "/" + p.get_directory() + "/" + p.get_name()

    def next_path_in_order(self):
        """:return the next images as the dataset loader would normally do"""
        p = self.imgDataArray[self.i]
        self.i += 1
        return self.baseDirectory + "/" + p.get_directory() + "/" + p.get_name()

    def next_path(self):
        """Return the path one image at each call in the order defined in the constructor
        It's independant from the normal iterator. Use this when you wanna go through the 
        dataset twice in a row"""
        p = self.imgDataArray[self.iterator_path]
        self.iterator_path += 1
        return self.baseDirectory + "/" + p.get_directory() + "/" + p.get_name()

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
                    info("DATASET LOADER]", "Loaded all imgs for training.")
                    self.train_loaded = False
                    self.i = 0
                    redo = False
                    break
                else:
                    info("DATASET LOADER]", "Loaded all imgs for training. Next call will load test data...")
                    redo = False
                    self.train_loaded = True
                    break
            if self.i == self.number_of_imgs:
                info("DATASET LOADER]", "Loaded all imgs for test. Done! Next call will load train data")
                redo = False
                self.train_loaded = False
                self.i = 0
                break
            if j + 1 >= self.max_img_loaded:
                info("DATASET LOADER]", "")
                print("Max img loaded!", self.i, "/", self.number_of_imgs)
                redo = True
                break

            img = cv2.imread(self.baseDirectory + "/" + self.imgDataArray[self.i].get_directory() + "/" +
                             self.imgDataArray[self.i].get_name(), cv2.IMREAD_COLOR)
            data_x.append(img)
            data_y.append(self.imgDataArray[self.i].get_img_class())
            j += 1
            self.i += 1

        info("DATASET LOADER]", "Loading completed!")

        return redo, np.asarray(data_x), np.asarray(data_y)

import os
import random
import cv2
import numpy as np
import math


#TODO GENERALISER la fonction à n layout
def split_list(a_list, number, layout):
    if layout == 0:
        split = int(len(a_list) / number)
        return a_list[:split], a_list[split:]
    elif layout == 1:
        split = int(len(a_list) / number)
        train1 = a_list[:split]
        test = a_list[split:2*split]
        train2 = a_list[2*split:]
        return test, train1 + train2
    elif layout == 2:
        split = int(len(a_list) / number)
        train1 = a_list[:2 * split]
        test = a_list[2 * split:3 * split]
        train2 = a_list[3 * split:]
        return test, train1 + train2
    elif layout == 3:
        split = int(len(a_list) / number)
        return a_list[3*split:], a_list[:3*split]


def shuffle_lists(a_list, b_list):
    list1_shuf = []
    list2_shuf = []
    index_shuf = list(range(len(a_list)))
    random.shuffle(index_shuf)
    for i in index_shuf:
        list1_shuf.append(a_list[i])
        list2_shuf.append(b_list[i])

    return list1_shuf, list2_shuf


class Image:
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


class ImgUtils:
    """ Image loading class into a readable format for keras.
        The folder layout must be as such:
        main Directory
        |_Classes Directory
          |_Classe files

    PARAMS:
        threshold: the number of image to load between trainings
        layout: paramètre entre 0 et 3 qui définit la position de la coupe dans le dataset
    """

    def __init__(self, base_directory, max_img_loaded):
        self.baseDirectory = base_directory
        self.max_img_loaded = max_img_loaded
        self.number_of_image_read = 0
        self.imgDataArray = []
        self.number_of_imgs = 0
        self.number_of_imgs_for_train = 0
        self.number_of_imgs_for_test = 0

        self.train_loaded = False

    def discover_and_make_order(self):
        print("Discovering dataset...")
        directories = next(os.walk(self.baseDirectory))[1]

        i = 0
        for directory in directories:
            for file_name in next(os.walk(self.baseDirectory + "/" + directory))[2]:
                self.imgDataArray.append(Image(file_name, directory, i))
                self.number_of_imgs += 1
            i += 1

        print(len(directories), "classes found.\n", self.number_of_imgs, "images found.")

        print("Shuffling order...")
        random.shuffle(self.imgDataArray)

        # TODO: faire de sorte d'être sur que on oublie pas une image ou deux
        self.number_of_imgs_for_train = math.floor((self.number_of_imgs/4)*3)
        self.number_of_imgs_for_test = math.floor(self.number_of_imgs/4)

        print("Ready for loading!\n", self.number_of_imgs_for_train, "for training and", self.number_of_imgs_for_test, "for testing")
        return len(directories)

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
            if j+1 >= self.max_img_loaded:
                print("Max img loaded!", self.number_of_image_read, "/", self.number_of_imgs)
                redo = True
                break

            img = cv2.imread(self.baseDirectory + "/" + img_data.get_directory() + "/" + img_data.get_name(), 0)
            img = np.expand_dims(img, axis=0)
            data_x.append(img)
            data_y.append(img_data.get_img_class())
            j += 1
            self.number_of_image_read += 1
            # print(img_data.get_name())

        print("Loading completed!")

        return redo, np.asarray(data_x), np.asarray(data_y)

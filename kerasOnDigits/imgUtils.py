import os
import random
import cv2
import numpy as np


def split_list(a_list, number, layout):
    print(layout)
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
        return a_list[split:], a_list[:split]


def shuffle_lists(a_list, b_list):
    list1_shuf = []
    list2_shuf = []
    index_shuf = list(range(len(a_list)))
    random.shuffle(index_shuf)
    for i in index_shuf:
        list1_shuf.append(a_list[i])
        list2_shuf.append(b_list[i])

    return list1_shuf, list2_shuf


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

    def __init__(self, base_directory, threshold):
        self.baseDirectory = base_directory
        self.already_computed = []
        self.threshold = threshold

    def load_and_shuffle_dataset(self, layout):

        redo = False
        directories = next(os.walk(self.baseDirectory))[1]
        print("Loading dataset.", len(directories), "classes found")
        data_x = []
        data_y = []

        i = 0
        j = 0
        for directory in directories:
            if any(directory in s for s in self.already_computed):
                continue

            files_names = next(os.walk(self.baseDirectory + "/" + directory))[2]
            for file_name in files_names:
                img = cv2.imread(self.baseDirectory + "/" + directory + "/" +file_name, 0)
                data_x.append(img)
                data_y.append(i)
                j += 1
                if j >= self.threshold:
                    break

            if j >= self.threshold:
                redo = True
                print("Threshold reached! Will load the rest of the data after the training...")
                break

            self.already_computed.append(directory)
            print("Class", i, "loaded")
            i += 1

        data_x, data_y = shuffle_lists(data_x, data_y)
        print("Loading completed!")

        test_x, train_x = split_list(data_x, 4, layout)
        test_y, train_y = split_list(data_y, 4, layout)

        return redo, len(directories), (np.asarray(train_x), np.array(train_y)), (np.asarray(test_x), np.array(test_y))

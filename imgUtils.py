import os
import random
import cv2
import numpy as np


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


class FolderReader:
    def __init__(self, files_array, id, directory_name):
        self.files_array = files_array
        self.counter = 0
        self.id = id
        self.directory_name = directory_name

    def has_next(self):
        return self.counter+1 <= len(self.files_array)

    def next(self):
        file = self.files_array[self.counter]
        self.counter += 1
        return file

    def get_id(self):
        return self.id

    def get_directory_name(self):
        return self.directory_name


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

    def load_and_shuffle_dataset(self, layout):

        redo = False
        directories = next(os.walk(self.baseDirectory))[1]
        print("Loading dataset.", len(directories), "classes found")

        folder_readers = []
        i = 0
        for directory in directories:
            folder_readers.append(FolderReader(next(os.walk(self.baseDirectory + "/" + directory))[2], i, directory))
            i += 1

        data_x = []
        data_y = []

        stop = False
        j = 0
        while not stop:
            stop = True
            for folder_reader in folder_readers:
                if not folder_reader.has_next():
                    continue
                if j+1 >= self.max_img_loaded :
                    print("Max img loaded! Will train an resume the loading after the training...")
                    redo = True
                    break
                stop = False

                img = cv2.imread(self.baseDirectory + "/" + folder_reader.get_directory_name() + "/" + folder_reader.next(), 0)
                img = np.expand_dims(img, axis=0)
                data_x.append(img)
                data_y.append(folder_reader.get_id())
                j += 1
                # print(folder_reader.get_id())

        print("Loading completed!")

        data_x, data_y = shuffle_lists(data_x, data_y)
        print("shuffle completed!")

        test_x, train_x = split_list(data_x, 4, layout)
        test_y, train_y = split_list(data_y, 4, layout)
        print("Split completed")

        return redo, len(directories), np.asarray(train_x), np.asarray(train_y), np.asarray(test_x), np.asarray(test_y)

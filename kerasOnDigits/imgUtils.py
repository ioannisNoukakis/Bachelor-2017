import os
import random
import cv2
import numpy as np

class ImgUtils:
    """ Image loading class into a readable format for keras.
        The folder layout must be as such:
        main Directory
        |_Classes Directory
          |_Classe files
    """

    def __init__(self, base_directory):
        self.baseDirectory = base_directory

    def load_and_shuffle_dataset(self):
        directories = next(os.walk(self.baseDirectory))[1]
        print("Loading dataset.", len(directories), "classes found")
        dataX = []
        dataY = []

        i = 0
        for directory in directories:
            files_names = next(os.walk(self.baseDirectory + "/" + directory))[2]
            for file_name in files_names:
                img = cv2.imread(self.baseDirectory + "/" + directory + "/" +file_name, 0)
                dataX.append(img)
                dataY.append(i)

            print("Class", i, "loaded")
            i += 1

        dataX, dataY = self.shuffle_lists(dataX, dataY)
        print("Loading completed!")

        testX, trainX = self.split_list(dataX, 4)
        testY, trainY = self.split_list(dataY, 4)

        return len(directories), (np.asarray(trainX), np.array(trainY)), (np.asarray(testX), np.array(testY))

    def split_list(self, a_list, number):
        split = int(len(a_list) / number)
        return a_list[:split], a_list[split:]

    def shuffle_lists(self, a_list, b_list):
        list1_shuf = []
        list2_shuf = []
        index_shuf = list(range(len(a_list)))
        random.shuffle(index_shuf)
        for i in index_shuf:
            list1_shuf.append(a_list[i])
            list2_shuf.append(b_list[i])

        return list1_shuf, list2_shuf

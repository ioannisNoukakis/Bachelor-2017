import os
import random
import cv2

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
        dataX = []
        dataY = []

        for directory in directories:
            files_names = next(os.walk(self.baseDirectory + "/" + directory))[2]
            # image = cv2.imread(argv[1], CV_LOAD_IMAGE_COLOR);
            for file_name in files_names:
                img = cv2.imread(self.baseDirectory + "/" + directory + "/" +file_name, 0)
                dataX.append(img)
                cv2.imshow('image', img)

        print(dataX)

        return len(directories),  # Returns the number of classes

    def split_list(a_list):
        half = len(a_list) / 2
        return a_list[:half], a_list[half:]

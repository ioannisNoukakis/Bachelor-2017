import cv2
import numpy as np
import os

from keras.preprocessing.image import img_to_array
from vis.utils import utils
from vis.utils.vggnet import VGG16
from main import get_model
from vis.visualization import visualize_cam


def main():

    model = get_model("custom", "GAP")
    print('Model loaded.')

    print(model.get_layer("W").get_weights()[0])
    print(len(model.get_layer("W").get_weights()[0]))


if __name__ == "__main__":
    main()

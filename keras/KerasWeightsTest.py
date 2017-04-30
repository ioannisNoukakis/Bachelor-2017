import cv2
import numpy as np
import os

from keras.preprocessing.image import img_to_array
from vis.utils import utils
from vis.utils.vggnet import VGG16
from vis.visualization import visualize_cam


def main():
    # Build the VGG16 network with ImageNet weights
    model = VGG16(weights='imagenet', include_top=True)
    print('Model loaded.')

    print(model.get_layer("block1_conv1").get_weights()[1])
    print(len(model.get_layer("block1_conv1").get_weights()[1]))


if __name__ == "__main__":
    main()

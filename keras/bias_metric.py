import csv
from glob import glob

import PIL
import keras
import psutil as psutil
from PIL import Image

import pyximport;
from sklearn.preprocessing import MinMaxScaler

pyximport.install()
from heatmapgenerate import *
import numpy as np


def compute_metric(cam: np.ndarray, mask: np.ndarray):
    inside = outside = 0.
    for i, row in enumerate(mask):
        for j, pixel in enumerate(row):

            if pixel >= 10:
                inside += cam[i][j]
            else:
                outside += cam[i][j]

    return outside/(inside + outside)


def compute_bias(b_directory, file_p, i, modifier:str=""):
    splitted = file_p.split('/')
    img_path = b_directory + '/' + splitted[-3] + '/' + splitted[-2] + '/' + i + '.tiff'
    mask_path = 'dataset_black_bg/' + splitted[-3] + '/' + splitted[-2][:-4] + '_final_masked.jpg'
    cam = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if modifier == 'normalizer01':
        cam = MinMaxScaler((0., 1.)).fit_transform(cam)
    if modifier == 'normalizerMin':
        cam -= cam.min()
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print('[USER WARNING]', mask_path, 'was not found. Check your file\'s names')
        return -1
    return compute_metric(cam, mask)

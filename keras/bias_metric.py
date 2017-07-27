import json
from threading import Thread

import cv2
import time
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def compute_metric(cam: np.ndarray, mask: np.ndarray):
    """
    Computes the bias metrics.

    :param cam: the cam
    :param mask: the mask (this is the subject surrounded by black background).
    :return: the bias score.
    """
    inside = outside = 0.
    for i, row in enumerate(mask):
        for j, pixel in enumerate(row):

            if pixel >= 10:
                inside += cam[i][j]
            else:
                outside += cam[i][j]

    return outside/(inside + outside)


def compute_bias(b_directory, file_p, i, modifier:str=""):
    """
    Computes bias for an image.

    :param b_directory: the base directory of the heatmap folder.
    :param file_p: the image path to the json file.
    :param i: the class to compute bias metric.
    :param modifier: pre processing over the input. Can be '', 'normalizer01' or 'normalizerMin'
    :return: The bias score.
    """
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


class BiasWorkerThread(Thread):
    """
    Multi threaded implementation of the bias metric algorithm.
    """
    def __init__(self, a, b, base_d, files_path):
        """
        :param a: starting index (index of the dataset)
        :param b: ending index (index of the dataset)
        :param base_d: the base directory of the heatmap folder.
        :param files_path: the image path to the json file.
        """
        Thread.__init__(self)
        self.a = a
        self.b = b
        self.files_path = files_path
        self.base_d = base_d

    def run(self):
        for i in range(self.a, self.b):
            start_time = time.time()
            try:
                with open(self.files_path[i]) as data_file:
                    data = json.load(data_file)
                if data['predicted'] == data['true_label']:
                    score = compute_bias(self.base_d, self.files_path[i], data['predicted'])
                    if score == -1:
                        continue
                    score_n01 = compute_bias(self.base_d, self.files_path[i], data['predicted'], 'normalizer01')
                    score_nmin = compute_bias(self.base_d, self.files_path[i], data['predicted'], 'normalizerMin')
                    with open(self.files_path[i], 'w') as outfile:
                        json.dump({'predicted': data['predicted'], "true_label": data['true_label'],
                                   'score': score, 'score_n01': score_n01, 'score_nmin': score_nmin}, outfile)
                else:
                    score_predicted = compute_bias(self.base_d, self.files_path[i], data['predicted'])
                    if score_predicted == -1:
                        continue
                    score_predicted_n01 = compute_bias(self.base_d, self.files_path[i], data['predicted'],
                                                       'normalizer01')
                    score_predicted_nmin = compute_bias(self.base_d, self.files_path[i], data['predicted'],
                                                        'normalizerMin')

                    score_true_label = compute_bias(self.base_d, self.files_path[i], data['true_label'])
                    score_true_label_n01 = compute_bias(self.base_d, self.files_path[i], data['true_label'],
                                                        'normalizer01')
                    score_true_label_nmin = compute_bias(self.base_d, self.files_path[i], data['true_label'],
                                                         'normalizerMin')

                    with open(self.files_path[i], 'w') as outfile:
                        json.dump({'predicted': data['predicted'], "true_label": data['true_label'],
                                   'score_predicted': score_predicted, 'score_predicted_n01': score_predicted_n01,
                                   'score_predicted_nmin': score_predicted_nmin,
                                   'score_true_label': score_true_label, 'score_true_label_n01': score_true_label_n01,
                                   'score_true_label_nmin': score_true_label_nmin},
                                  outfile)
            except (KeyError, json.decoder.JSONDecodeError):
                print('[USER WARNING]', 'Json was malformed. Perhaps you cam generation was interrupted?')
            print("ok(", time.time() - start_time, ") seconds")
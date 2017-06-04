import csv

import PIL
import keras
from PIL import Image

from img_loader import DatasetLoader
from img_processing import merge_images_mask, most_dominant_color, color_distance
from model_utils import get_heatmap


class BiasMetric:

    def __init__(self):
        self.l1 = []  # l1 is the progress of the fine tuned algorithm
        self.l2 = []  # l2 is the progress of the custom algorithm
        self.e1 = []  # l2 is the progress of the error of the custom algorithm
        self.e2 = []  # l2 is the progress of the error of the custom algorithm

        self.metric1 = []  # metric1 is l1 - l2
        self.metric2 = []  # metric1 is e1 - e2

    def save_to_csv(self):
        data = []
        for i, _ in enumerate(self.l1):
            data.append([self.l1[i], self.l2[i], self.e1[i], self.e2[i], self.metric1[i], self.metric2[i]])

        with open('results.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerows(data)


class MetricCallback(keras.callbacks.Callback):

    def __init__(self, bias_metric: BiasMetric,
                 dummy_model,
                 batch_size,
                 current_loader: DatasetLoader,
                 segmentend_db_img_loader: DatasetLoader):

        self.bias_metric = bias_metric
        self.dummy_model = dummy_model
        self.batch_size = batch_size
        self.current_loader = current_loader
        self.segmentend_db_img_loader = segmentend_db_img_loader

    def on_batch_end(self, batch, logs={}):
        for i in range(0, self.batch_size):
            # load images
            if not self.current_loader.has_next():
                return
            training_img = Image.open(self.current_loader.next_path())
            mask = Image.open(self.segmentend_db_img_loader.next_path_in_order())

            # get the cams from both the models
            cam_a = get_heatmap(training_img, self.model, 'block5_conv3')
            cam_b = get_heatmap(training_img, self.dummy_model, 'Conv4')

            # apply the segmented db mask
            mask = mask.resize((224, 224), PIL.Image.ANTIALIAS)
            cam_a_p, cam_a_e = merge_images_mask(cam_a, mask)
            cam_b_p, cam_b_e = merge_images_mask(cam_b, mask)

            # get the dominant color
            a_p_dc = most_dominant_color(cam_a_p)
            a_e_dc = most_dominant_color(cam_a_e)
            b_p_dc = most_dominant_color(cam_b_p)
            b_e_dc = most_dominant_color(cam_b_e)

            # get the distance from red
            l1 = color_distance((255, 0, 0), a_p_dc)
            l2 = color_distance((0, 0, 255), b_p_dc)
            e1 = color_distance((0, 0, 255), a_e_dc)
            e2 = color_distance((0, 0, 255), b_e_dc)
            self.bias_metric.l1.append(l1)
            self.bias_metric.l2.append(l2)
            self.bias_metric.e1.append(e1)
            self.bias_metric.e2.append(e2)

            # compute the metrics
            self.bias_metric.metric1.append(l1 - l2)
            self.bias_metric.metric2.append(e1 - e2)
            print("image", i, "ok")

        return

import csv
import time
from threading import Thread

import PIL
import keras
from PIL import Image
from keras.engine import Model

from img_loader import DatasetLoader
import img_processing.img_processing
import img_processing.heatmap_generate
from multiprocessing.pool import ThreadPool


class BiasMetric:

    def __init__(self, graph_context):
        self.l1 = []  # l1 is the progress of the fine tuned algorithm
        self.l2 = []  # l2 is the progress of the custom algorithm
        self.e1 = []  # l2 is the progress of the error of the custom algorithm
        self.e2 = []  # l2 is the progress of the error of the custom algorithm

        self.graph_context = graph_context

        self.metric1 = []  # metric1 is l1 - l2
        self.metric2 = []  # metric1 is e1 - e2

    def save_to_csv(self):
        data = []
        for i, _ in enumerate(self.l1):
            data.append([self.l1[i], self.l2[i], self.e1[i], self.e2[i], self.metric1[i], self.metric2[i]])

        with open('results.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerows(data)


class HeatmapCompute(Thread):

    def __init__(self, img, model, layer_name):
        Thread.__init__(self)
        self.img = img
        self.model = model
        self.layer_name = layer_name
        self._return = None

    def run(self):
        self._return = img_processing.heatmap_generate.heatmap_generate(self.img, self.model, self.layer_name)

    def join(self):
        Thread.join(self)
        return self._return


class MetricCallback(keras.callbacks.Callback):

    def __init__(self, bias_metric: BiasMetric,
                 dummy_model: Model,
                 shampleing_rate: int,
                 current_loader: DatasetLoader,
                 segmentend_db_img_loader: DatasetLoader):
        """

        :param bias_metric: The bias metrics container
        :param dummy_model: The dummy model that wont be trained.
        :param shampleing_rate: The every n images a metric will be comptued
        :param current_loader: The dataset loader
        :param segmentend_db_img_loader: the mask dataset loader.

        """

        self.bias_metric = bias_metric
        self.dummy_model = dummy_model
        self.shampleing_rate = shampleing_rate
        self.i = 0
        self.j = 0
        self.current_loader = current_loader
        self.segmentend_db_img_loader = segmentend_db_img_loader

    def on_batch_end(self, batch, logs={}):
        for _ in range(0, 10):
            if self.i == self.shampleing_rate:
                # load images
                if not self.current_loader.has_next():
                    return

                print("Starting image", self.j)
                start_time = time.time()

                training_img = Image.open(self.current_loader.next_path())
                mask = Image.open(self.segmentend_db_img_loader.next_path_in_order())

                # get the cams from both the models
                pool = ThreadPool(processes=2)
                async_result1 = pool.apply_async(img_processing.heatmap_generate.heatmap_generate, (self.bias_metric.graph_context,
                                                                                                    training_img,
                                                                                                    self.model,
                                                                                                    'block5_conv3'))
                async_result2 = pool.apply_async(img_processing.heatmap_generate.heatmap_generate, (self.bias_metric.graph_context,
                                                                                                    training_img,
                                                                                                    self.dummy_model,
                                                                                                    'Conv4'))

                cam_a = async_result1.get()
                cam_b = async_result2.get()

                print("got cams in", time.time() - start_time)
                start_time = time.time()

                # apply the segmented db mask
                mask = mask.resize((224, 224), PIL.Image.ANTIALIAS)
                cam_a_p, cam_a_e = img_processing.img_processing.merge_images_mask(cam_a, mask)
                cam_b_p, cam_b_e = img_processing.img_processing.merge_images_mask(cam_b, mask)

                print("mask applied in", time.time() - start_time)
                start_time = time.time()

                # get the distance from red
                l1 = img_processing.img_processing.pixels_counter(cam_a_p, (255, 0, 0), (183, 253, 52))
                l2 = img_processing.img_processing.pixels_counter(cam_b_p, (255, 0, 0), (183, 253, 52))
                e1 = img_processing.img_processing.pixels_counter(cam_a_e, (255, 0, 0), (183, 253, 52))
                e2 = img_processing.img_processing.pixels_counter(cam_b_e, (255, 0, 0), (183, 253, 52))

                print("pixels computed in ", time.time() - start_time)

                self.bias_metric.l1.append(l1)
                self.bias_metric.l2.append(l2)
                self.bias_metric.e1.append(e1)
                self.bias_metric.e2.append(e2)

                # compute the metrics
                self.bias_metric.metric1.append(l1 - l2)
                self.bias_metric.metric2.append(e1 - e2)
                self.i = 0
            else:
                self.i += 1
            self.j += 1
        return

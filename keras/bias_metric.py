import csv
import os
import time
from pathlib import Path
from threading import Thread

import PIL
import keras
import psutil as psutil

from PIL import Image
from keras.engine import Model

from img_loader import DatasetLoader
import img_processing.img_processing
import img_processing.heatmap_generate
from multiprocessing.pool import ThreadPool

from logger import info, error


class BiasMetric:
    """Class used as container for metrics."""

    def __init__(self, graph_context):
        self.l1 = []  # l1 is the progress of the fine tuned algorithm
        self.l2 = []  # l2 is the progress of the custom algorithm
        self.e1 = []  # l2 is the progress of the error of the custom algorithm
        self.e2 = []  # l2 is the progress of the error of the custom algorithm

        self.graph_context = graph_context

        self.metric1 = []  # metric1 is l1 - l2
        self.metric2 = []  # metric1 is e1 - e2

    def save_to_csv(self):
        """
        Save the experiments results on a csv file.
        :return:-
        """
        data = [['l1', 'l2', 'e1', 'e2']]
        for i, _ in enumerate(self.l1):
            data.append([self.l1[i], self.l2[i], self.e1[i], self.e2[i]])

        with open('results.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerows(data)


def get_mem_usage():
    """
    Get the current memory usage of this device.
    :return: the memory info.
    """
    process = psutil.Process(os.getpid())
    return process.memory_info()


class HeatmapCompute(Thread):
    """
    Worker thread to compute Heatmaps (CAMs)
    """
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


class MonoMetricCallBack(keras.callbacks.Callback):
    """
    Simple bias metric.
    """

    def __init__(self, bias_metric: BiasMetric,
                 shampleing_rate: int,
                 current_loader: DatasetLoader):
        """

        :param bias_metric: The bias metrics container
        :param shampleing_rate: The every n images a metric will be comptued
        :param current_loader: The dataset loader

        """

        self.bias_metric = bias_metric
        self.shampleing_rate = shampleing_rate
        self.i = 0
        self.j = 0
        self.k = 0
        self.current_loader = current_loader

    def on_epoch_end(self, epoch, logs=None):
        self.i = 0
        self.j = 0

    def on_batch_end(self, batch, logs={}):
        print(self.i, self.j)
        for _ in range(0, 10):  # the cnn takes 10 by 10 images
            if self.i == self.shampleing_rate:

                # check boundary
                if self.j >= self.current_loader.number_of_imgs:
                    break
                info("[BIAS METRIC][Memory]", get_mem_usage())

                info("[BIAS METRIC]", "")
                print("Starting image", self.k)
                start_time = time.time()
                self.k += 1

                p = self.current_loader.get(self.j)
                p_file = Path(p)
                if not p_file.exists():  # if segmented does not exists continue...
                    error("[ERROR][BIAS METRIC] -> does not exists:", p)
                    self.j += 1
                    continue
                training_img = Image.open(p)

                tmp = p[:-4]
                tmp = tmp[len(self.current_loader.baseDirectory):]
                tmp = "./segmentedDB" + tmp + "_final_masked.jpg"
                print(tmp)

                tmp_file = Path(tmp)
                if not tmp_file.exists():  # if segmented does not exists continue...
                    error("[ERROR][BIAS METRIC] -> does not exists:", tmp)
                    self.j += 1
                    continue
                mask = Image.open(tmp)

                # get the cams from both the models
                pool = ThreadPool(processes=2)
                async_result1 = pool.apply_async(img_processing.heatmap_generate.heatmap_generate,
                                                 (self.bias_metric.graph_context,
                                                  training_img,
                                                  self.model,
                                                  'block5_conv3'))

                cam_a = async_result1.get()

                if cam_a is None:
                    print("[ERROR][BIAS METRIC] -> could not read this image:", tmp)
                    self.j += 1
                    continue

                print("got cams in", time.time() - start_time)
                start_time = time.time()

                # apply the segmented db mask
                mask = mask.resize((224, 224), PIL.Image.ANTIALIAS)
                cam_a_p, cam_a_e = img_processing.img_processing.merge_images_mask(cam_a, mask)

                print("mask applied in", time.time() - start_time)
                start_time = time.time()

                # get the red pixels ratio
                l1 = img_processing.img_processing.pixels_counter(cam_a_p, (255, 0, 0), (183, 253, 52))
                e1 = img_processing.img_processing.pixels_counter(cam_a_e, (255, 0, 0), (183, 253, 52))

                print("pixels computed in ", time.time() - start_time)

                self.bias_metric.l1.append(l1)
                self.bias_metric.l2.append(0)
                self.bias_metric.e1.append(e1)
                self.bias_metric.e2.append(0)

                self.i = 0
            else:
                self.i += 1
            self.j += 1



class MetricCallback(keras.callbacks.Callback):
    """
    Double bias metrics. One this the network and one with randomly initialized model
    """

    def __init__(self, bias_metric: BiasMetric,
                 dummy_model: Model,
                 shampleing_rate: int,
                 current_loader: DatasetLoader):
        """

        :param bias_metric: The bias metrics container
        :param dummy_model: The dummy model that wont be trained.
        :param shampleing_rate: The every n images a metric will be comptued
        :param current_loader: The dataset loader

        """

        self.bias_metric = bias_metric
        self.dummy_model = dummy_model
        self.shampleing_rate = shampleing_rate
        self.i = 0
        self.j = 0
        self.current_loader = current_loader

    def on_batch_end(self, batch, logs={}):
        for _ in range(0, 10):  # the cnn takes 10 by 10 images
            if self.i == self.shampleing_rate:

                # check boundary
                if self.j > self.current_loader.number_of_imgs:
                    return

                print("[INFO][BIAS METRIC]", "[Memory]", get_mem_usage())

                print("Starting image", self.j)
                start_time = time.time()

                # check if image exists
                p = self.current_loader.get(self.j)
                p_file = Path(p)
                if not p_file.exists():  # if segmented does not exists continue...
                    print("[ERROR][BIAS METRIC]", p, "does not exists...")
                    self.j += 1
                    continue
                training_img = Image.open(p)

                tmp = p[:-4]
                tmp = tmp[len(self.current_loader.baseDirectory):]
                tmp = "./segmentedDB" + tmp + "_final_masked.jpg"
                print(tmp)

                # check if image exists
                tmp_file = Path(tmp)
                if not tmp_file.exists():  # if segmented does not exists continue...
                    print("[ERROR][BIAS METRIC]", tmp, "does not exists...")
                    self.j += 1
                    continue
                mask = Image.open(tmp)

                # get the cams from both the models
                pool = ThreadPool(processes=2)
                async_result1 = pool.apply_async(img_processing.heatmap_generate.heatmap_generate,
                                                 (self.bias_metric.graph_context,
                                                  training_img,
                                                  self.model,
                                                  'block5_conv3'))
                async_result2 = pool.apply_async(img_processing.heatmap_generate.heatmap_generate,
                                                 (self.bias_metric.graph_context,
                                                  training_img,
                                                  self.dummy_model,
                                                  'Conv4'))

                cam_a = async_result1.get()
                cam_b = async_result2.get()

                if cam_a is None or cam_b is None:
                    print("[ERROR][BIAS METRIC]", "could not read this image:", tmp)
                    self.j += 1
                    continue

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

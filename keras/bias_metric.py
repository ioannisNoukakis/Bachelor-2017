import csv
import gc
import os
import time
from multiprocessing.pool import ThreadPool
from pathlib import Path
from threading import Thread

import PIL
import img_processing
import keras
import psutil as psutil
from PIL import Image

from heatmapgenerate import heatmap_generate
from img_loader import DatasetLoader
from logger import info, error


def save_to_csv(l, e):
    """
    Save the experiments results on a csv file.
    :return:-
    """

    with open('results.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow([l, e])
        f.close()


class BiasMetric:
    """Class used as container for metrics."""

    def __init__(self, graph_context):
        self.graph_context = graph_context


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
        self._return = heatmap_generate.heatmap_generate(self.img, self.model, self.layer_name)

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
        self.pool = ThreadPool(processes=2)

    def on_train_begin(self, logs=None):
        save_to_csv('l', 'e')

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

                async_result1 = self.pool.apply_async(heatmap_generate,
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

                save_to_csv(l1, e1)
                self.i = 0
            else:
                self.i += 1
            self.j += 1
            gc.collect()


def compute_metric(cam, mask):
    # apply the segmented db mask
    mask = mask.resize((224, 224), PIL.Image.ANTIALIAS)
    cam_a_p, cam_a_e = img_processing.merge_images_mask(cam, mask)

    # get the red pixels ratio
    # FIXME do a scale and attribute scores
    l1 = img_processing.pixels_counter(cam_a_p, (255, 0, 0), (183, 253, 52))
    e1 = img_processing.pixels_counter(cam_a_e, (255, 0, 0), (183, 253, 52))

    save_to_csv(l1, e1)

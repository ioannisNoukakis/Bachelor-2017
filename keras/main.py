import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Thread

import tensorflow as tf
from PIL import Image
from keras.models import load_model

from VGG16_ft import VGG16FineTuned
from bias_metric import compute_metric
from img_processing import dataset_convertor
from plant_village_custom_model import *
import random
from numpy import argmax
from keras.applications.imagenet_utils import preprocess_input
import uuid
import time
import pyximport; pyximport.install()
from heatmapgenerate import *


# https://elitedatascience.com/keras-tutorial-deep-learning-in-python#step-1
# http://cnnlocalization.csail.mit.edu/
# https://arxiv.org/pdf/1312.4400.pdf
# https://www.quora.com/What-is-global-average-pooling
# https://arxiv.org/pdf/1512.04150.pdf
# https://arxiv.org/pdf/1512.03385.pdf
# http://lcn.epfl.ch/tutorial/english/perceptron/html/learning.html
# https://github.com/fchollet/keras/issues/4446

"""
def create_cam(model, outname, viz_folder, layer_name):

    heatmaps = []
    for path in next(os.walk(viz_folder))[2]:
        # Predict the corresponding class for use in `visualize_saliency`.
        seed_img = utils.load_img(viz_folder + '/' + path, target_size=(256, 256))

        # Here we are asking it to show attention such that prob of `pred_class` is maximized.
        heatmap = img_processing.heatmap_generate.heatmap_generate(seed_img, model, layer_name, None, True)
        heatmaps.append(heatmap)

    cv2.imwrite(outname, utils.stitch_images(heatmaps))


def make_simple_bias_metrics(dataset_name: str, shampeling_rate: int):

    info("[INFO][MAIN]", "Loading...")
    dataset_loader = DatasetLoader(dataset_name, 10000)

    info("[INFO][MAIN]", "Compiling model...")
    vgg16 = VGG16FineTuned(dataset_loader)
    graph_context = tf.get_default_graph()

    bias_metric = BiasMetric(graph_context)
    mc = MonoMetricCallBack(bias_metric=bias_metric,
                            shampleing_rate=shampeling_rate,
                            current_loader=dataset_loader)

    info("[INFO][MAIN]", "Starting training...")
    vgg16.train(10, False, [mc])

    info("[INFO][MAIN]", "Training completed!")"""


def generate_maps(context, dl: DatasetLoader, model, map_out: str, begining_index: int, end_index: int, number: int):
    with context.as_default():
        tmp_name = uuid.uuid1().hex
        # plot CAMs only for the validation data:
        for i in range(begining_index, end_index):
            outpath = map_out + "/" + dl.imgDataArray[i].directory + "/" + dl.imgDataArray[i].name
            try:
                os.makedirs(outpath)
            except OSError:
                print("CAMS already done... skipping...")
                continue
            for j in range(0, dl.nb_classes):
                try:
                    outname = outpath + "/" + str(j) + ".png"

                    img = cv2.imread(dl.baseDirectory + "/" + dl.imgDataArray[i].directory + "/" +
                                     dl.imgDataArray[i].name, cv2.IMREAD_COLOR)
                    predict_input = np.expand_dims(img, axis=0)
                    predict_input = predict_input.astype('float32')
                    predict_input = preprocess_input(predict_input)
                    predictions = model.predict(predict_input)
                    value = argmax(predictions)
                    start_time = time.time()
                    heatmap = heatmap_generate(
                        input_img=predict_input[0],
                        model=model,
                        class_to_predict=j,
                        layer_name='CAM',
                        tmp_name=tmp_name)
                    heatmap.save(outname)
                    print("got cams in", time.time() - start_time)
                    with open(outpath + '/resuts.json', 'w') as outfile:
                        json.dump({'predicted': str(value), "true_label": str(dl.imgDataArray[i].img_class)}, outfile)
                except:
                    print("ERROR IN THREAD", number, "PASSING...")


class MapWorker(Thread):
    def __init__(self, context, dl: DatasetLoader, model, map_out: str, begining_index: int, end_index: int,
                 number: int):
        super().__init__()
        self.context = context
        self.dl = dl
        self.model = model
        self.map_out = map_out
        self.begining_index = begining_index
        self.end_index = end_index
        self.number = number

    def run(self):
        with self.context.as_default():
            print("Thread", self.number, "started...")
            generate_maps(self.context, self.dl, self.model, self.map_out, self.begining_index, self.end_index, self.number)


def main():
    np.random.seed(123)  # for reproducibility
    random.seed(123)

    argv = sys.argv
    if argv[1] == "0":
        dl = DatasetLoader(argv[2], 10000)
        print("SEED IS", 123)
        print(dl.imgDataArray[dl.number_of_imgs_for_train].name)
        print(dl.imgDataArray[dl.number_of_imgs_for_train + 1].name)
        vggft = VGG16FineTuned(dataset_loader=DatasetLoader(argv[2], 10000), mode=argv[4])
        vggft.train(int(argv[5]), weights_out=argv[3])
    # ==================================================================================================
    if argv[1] == "1": # FIXME Cythonize
        numberOfCors = int(argv[2])
        dl = DatasetLoader(argv[3], 10000)
        model = load_model(argv[4])
        model._make_predict_function()  # have to initialize before threading
        graph = tf.get_default_graph()
        nb_to_process = dl.number_of_imgs_for_test
        inc = int(nb_to_process / numberOfCors)
        b_index = dl.number_of_imgs_for_train
        e_index = dl.number_of_imgs_for_train + inc
        print("images to process:", nb_to_process)
        print("inc is:", inc)
        print(numberOfCors, "workers will rise")
        threads = []
        if argv[5] == "thread":
            for i in range(0, numberOfCors):
                t = MapWorker(context=graph,
                              dl=dl,
                              model=model,
                              map_out=argv[6],
                              begining_index=b_index,
                              end_index=e_index,
                              number=i)
                t.start()
                threads.append(t)
                print(b_index, e_index)
                b_index = e_index
                e_index += inc
                time.sleep(2)
            for t in threads:
                t.join()
        else:
            generate_maps(dl=dl,
                          model=model,
                          map_out=argv[5],
                          begining_index=b_index,
                          end_index=e_index)

    if argv[1] == '2':
        dl = DatasetLoader(argv[3], 10000)
        model = load_model(argv[2])

        for i in range(dl.number_of_imgs_for_train, dl.number_of_imgs):
            outpath = argv[3] + "/" + dl.imgDataArray[i].directory + "/" + dl.imgDataArray[i].name
            heatmap_path = outpath + "/" + str(dl.imgDataArray[i].img_class) + ".png"

            p_file = Path(heatmap_path)
            if not p_file.exists():  # if segmented does not exists continue...
                print("[ERROR][BIAS METRIC] -> does not exists:", heatmap_path)
                continue
            heatmap = Image.open(heatmap_path)

            tmp = heatmap_path[:-4]
            tmp = tmp[len(argv[3]):]
            tmp = "./dataset_black_bg" + tmp + "_final_masked.jpg"
            print(tmp)

            tmp_file = Path(tmp)
            if not tmp_file.exists():  # if segmented does not exists continue...
                print("[ERROR][BIAS METRIC] -> does not exists:", tmp)
                continue
            mask = Image.open(tmp)

            compute_metric(heatmap, mask)
    if argv[1] == "3":
        dataset_convertor('dataset_black_bg', 'dataset_rand', 'dataset_art')
    if argv[1] == "4":
        directories = next(os.walk(argv[2]))[1]
        directories = sorted(directories)
        i = 0
        for directory in directories:
            for _ in next(os.walk(argv[2] + "/" + directory))[1]:
                i += 1
        print(i, "images processed.")


if __name__ == "__main__":
    main()

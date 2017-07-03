import sys
from vis.utils import utils

import img_processing.heatmap_generate
from CAM_maker import train_VGGCAM, plot_classmap
from VGG16_ft import VGG16FineTuned
from bias_metric import BiasMetric, MonoMetricCallBack
from plant_village_custom_model import *

import json

import tensorflow as tf
import os


# https://elitedatascience.com/keras-tutorial-deep-learning-in-python#step-1
# http://cnnlocalization.csail.mit.edu/
# https://arxiv.org/pdf/1312.4400.pdf
# https://www.quora.com/What-is-global-average-pooling
# https://arxiv.org/pdf/1512.04150.pdf
# https://arxiv.org/pdf/1512.03385.pdf


def create_cam(model, outname, viz_folder, layer_name):
    """
    create an image of Class Activation Mapping (CAM).

    :param model: The model
    :param outname: The name of the future generated image.
    :param viz_folder: The folder of the input images
    :param layer_name: The name of the layer of which the outputs will be used to compute the CAMs.
    :return: -
    """
    heatmaps = []
    for path in next(os.walk(viz_folder))[2]:
        # Predict the corresponding class for use in `visualize_saliency`.
        seed_img = utils.load_img(viz_folder + '/' + path, target_size=(256, 256))

        # Here we are asking it to show attention such that prob of `pred_class` is maximized.
        heatmap = img_processing.heatmap_generate.get_heatmap(seed_img, model, layer_name, None, True)
        heatmaps.append(heatmap)

    cv2.imwrite(outname, utils.stitch_images(heatmaps))


def make_simple_bias_metrics(dataset_name: str, shampeling_rate: int):
    """
    Make the bias metrics by using the process described here:
    <insert link to TB>

    :param dataset_name: The dataset name
    :param shampeling_rate: images will be processed every n image.
    :return: -
    """
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

    info("[INFO][MAIN]", "Training completed!")


def main():
    np.random.seed(123)  # for reproducibility

    argv = sys.argv
    if argv[1] == "0":
        vggft = VGG16FineTuned(dataset_loader=DatasetLoader(argv[2], 10000))
        vggft.train(15, weights_out=argv[3])
    if argv[1] == "1":
        dl = DatasetLoader(argv[2], 10000, True)
        train_VGGCAM(dl, int(argv[3]))

        # plot CAMs only for the validation data:
        for i in range(dl.number_of_imgs_for_train, dl.number_of_imgs):
            for j in range(0, dl.nb_classes):
                outpath = "maps/" + dl.imgDataArray[i].directory + "/" + dl.imgDataArray[i].name
                outname = outpath + "/" + str(j) + ".png"

                try:
                    os.makedirs("maps/" + dl.imgDataArray[i].directory + "/" + dl.imgDataArray[i].name)
                except OSError:
                    pass

                predicted = plot_classmap(outname=outpath + outname,
                              img_path=dl.baseDirectory + "/" +dl.imgDataArray[i].directory + "/" + dl.imgDataArray[i].name,
                              label=j,
                              nb_classes=dl.nb_classes)
                with open(outpath + '/resuts.json', 'w') as outfile:
                    json.dump({'predicted': predicted, "true_label": dl.imgDataArray[i].img_class}, outfile)

if __name__ == "__main__":
    main()

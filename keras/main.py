import sys
from vis.utils import utils

import img_processing.heatmap_generate
from VGG16_ft import VGG16FineTuned
from bias_metric import BiasMetric, MetricCallback, MonoMetricCallBack
from plant_village_custom_model import *

from img_processing.img_processing import dataset_convertor

import tensorflow as tf


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
    save_to_csv()


def make_bias_metrics(dataset_name: str, shampeling_rate: int):
    """
    Make the bias metrics by using the process described here:
    <insert link to TB>

    :param dataset_name: The dataset name
    :param shampeling_rate: images will be processed every n image.
    :return: -
    """
    img_u = DatasetLoader(dataset_name, 10000)

    dummy_model = get_custom_model(img_u, "GAP", random=True)
    vgg16 = VGG16FineTuned(img_u)
    graph_context = tf.get_default_graph()

    bias_metric = BiasMetric(graph_context)
    mc = MetricCallback(bias_metric=bias_metric,
                        dummy_model=dummy_model,
                        shampleing_rate=shampeling_rate,
                        current_loader=img_u)

    vgg16.train(5, False, [mc])

    save_to_csv()


def main():
    np.random.seed(123)  # for reproducibility

    argv = sys.argv
    if argv[1] == "0":
        make_simple_bias_metrics(argv[2], int(argv[3]))
    if argv[1] == "1":
        make_bias_metrics(argv[2], int(argv[3]))
    if argv[1] == "2":
        model = get_custom_model(DatasetLoader("dataset", 10000), "GAP", save="Custom_normal")
        create_cam(model, "CAM_normal_normal.jpg", "visual", "Conv4")
        create_cam(model, "CAM_normal_art.jpg", "visual_art", "Conv4")
        create_cam(model, "CAM_normal_rand.jpg", "visual_rand", "Conv4")
    if argv[1] == "3":
        model = get_custom_model(DatasetLoader("dataset_art", 10000), "GAP", save="Custom_art")
        create_cam(model, "CAM_art_normal.jpg", "visual", "Conv4")
        create_cam(model, "CAM_art_art.jpg", "visual_art", "Conv4")
        create_cam(model, "CAM_art_rand.jpg", "visual_rand", "Conv4")
    if argv[1] == "4":
        model = get_custom_model(DatasetLoader("dataset_art", 10000), "GAP", save="Custom_rand")
        create_cam(model, "CAM_rand_normal.jpg", "visual", "Conv4")
        create_cam(model, "CAM_rand_art.jpg", "visual_art", "Conv4")
        create_cam(model, "CAM_rand_rand.jpg", "visual_rand", "Conv4")
    if argv[1] == "5":
        model = get_custom_model(DatasetLoader("segmentedDB", 10000), "GAP", save="Custom_segmented")
        create_cam(model, "CAM_seg_normal.jpg", "visual", "Conv4")
        create_cam(model, "CAM_seg_art.jpg", "visual_art", "Conv4")
        create_cam(model, "CAM_seg_rand.jpg", "visual_rand", "Conv4")
    if argv[1] == "6":
        model = VGG16FineTuned(DatasetLoader("dataset", 10000))
        model.train(10, False, None)
        create_cam(model, "CAM_FT_normal_normal.jpg", "visual", "block5_conv3")
        create_cam(model, "CAM_FT_normal_art.jpg", "visual_art", "block5_conv3")
        create_cam(model, "CAM_FT_normal_rand.jpg", "visual_rand", "block5_conv3")
    if argv[1] == "7":
        model = VGG16FineTuned(DatasetLoader("dataset_art", 10000))
        model.train(10, False, None)
        create_cam(model, "CAM_FT_art_normal.jpg", "visual", "block5_conv3")
        create_cam(model, "CAM_FT_art_art.jpg", "visual_art", "block5_conv3")
        create_cam(model, "CAM_FT_art_rand.jpg", "visual_rand", "block5_conv3")
    if argv[1] == "8":
        model = VGG16FineTuned(DatasetLoader("dataset_random", 10000))
        model.train(10, False, None)
        create_cam(model, "CAM_FT_rand_normal.jpg", "visual", "block5_conv3")
        create_cam(model, "CAM_FT_rand_art.jpg", "visual_art", "block5_conv3")
        create_cam(model, "CAM_FT_rand_rand.jpg", "visual_rand", "block5_conv3")
    if argv[1] == "9":
        model = VGG16FineTuned(DatasetLoader("segmentedDB", 10000))
        model.train(10, False, None)
        create_cam(model, "CAM_FT_seg_normal.jpg", "visual", "block5_conv3")
        create_cam(model, "CAM_FT_seg_art.jpg", "visual_art", "block5_conv3")
        create_cam(model, "CAM_FT_seg_rand.jpg", "visual_rand", "block5_conv3")
    if argv[1] == "10":
        dataset_convertor("segmentedDB", "dataset_random", "dataset_art")


if __name__ == "__main__":
    main()

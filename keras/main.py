import sys
from vis.utils import utils

from VGG16_ft import VGG16FineTuned
from bias_metric import BiasMetric, MetricCallback
from plant_village_custom_model import *
from model_utils import get_heatmap
from img_processing import dataset_convertor



# https://elitedatascience.com/keras-tutorial-deep-learning-in-python#step-1
# http://cnnlocalization.csail.mit.edu/
# https://arxiv.org/pdf/1312.4400.pdf
# https://www.quora.com/What-is-global-average-pooling
# https://arxiv.org/pdf/1512.04150.pdf
# https://arxiv.org/pdf/1512.03385.pdf


def create_cam(model, outname, viz_folder):
    heatmaps = []
    for path in next(os.walk(viz_folder))[2]:
        # Predict the corresponding class for use in `visualize_saliency`.
        seed_img = utils.load_img(viz_folder + '/' + path, target_size=(256, 256))

        # Here we are asking it to show attention such that prob of `pred_class` is maximized.
        heatmap = get_heatmap(seed_img, model, "Conv4", None, True)
        heatmaps.append(heatmap)

    cv2.imwrite(outname, utils.stitch_images(heatmaps))


def make_bias_metrics():
    img_u = DatasetLoader("./dataset", 10000)
    img_segmented = DatasetLoader("./segmentedDB", 10000)

    for p in img_segmented.imgDataArray:
        p.name[:-4] += "_final_masked.jpg"

    bias_metric = BiasMetric()
    dummy_model = get_custom_model("GAP", True)
    vgg16 = VGG16FineTuned(img_u)
    mc = MetricCallback(bias_metric, dummy_model, 10, img_u, img_segmented)

    vgg16.train(1, False, [mc])

    bias_metric.save_to_csv()


def main():
    np.random.seed(123)  # for reproducibility

    argv = sys.argv
    if argv[1] == "1":
        make_bias_metrics()
    if argv[1] == "2":
        model = get_custom_model("GAP", "dataset", save="Custom_normal")
        create_cam(model, "CAM_normal_normal.jpg", "visual")
        create_cam(model, "CAM_normal_art.jpg", "visual_art")
        create_cam(model, "CAM_normal_rand.jpg", "visual_rand")
    if argv[1] == "3":
        model = get_custom_model("GAP", "dataset_art", save="Custom_art")
        create_cam(model, "CAM_art_normal.jpg", "visual")
        create_cam(model, "CAM_art_art.jpg", "visual_art")
        create_cam(model, "CAM_art_rand.jpg", "visual_rand")
    if argv[1] == "4":
        model = get_custom_model("GAP", "dataset_random", save="Custom_rand")
        create_cam(model, "CAM_rand_normal.jpg", "visual")
        create_cam(model, "CAM_rand_art.jpg", "visual_art")
        create_cam(model, "CAM_rand_rand.jpg", "visual_rand")
    if argv[1] == "5":
        dataset_convertor("segmentedDB", "dataset_random", "dataset_art")


if __name__ == "__main__":
    main()

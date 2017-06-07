from vis.utils import utils

from VGG16_ft import VGG16FineTuned
from bias_metric import BiasMetric, MetricCallback
from plant_village_custom_model import *
import tensorflow as tf


# https://elitedatascience.com/keras-tutorial-deep-learning-in-python#step-1
# http://cnnlocalization.csail.mit.edu/
# https://arxiv.org/pdf/1312.4400.pdf
# https://www.quora.com/What-is-global-average-pooling
# https://arxiv.org/pdf/1512.04150.pdf
# https://arxiv.org/pdf/1512.03385.pdf


def create_cam():
    model = get_custom_model("custom", "GAP")
    heatmaps = []
    for path in next(os.walk('./visual'))[2]:
        # Predict the corresponding class for use in `visualize_saliency`.
        seed_img = utils.load_img('./visual/' + path, target_size=(256, 256))

        # Here we are asking it to show attention such that prob of `pred_class` is maximized.
        # heatmap = heatmap_generate(seed_img, model, "Conv4")
        heatmaps.append(heatmap)

    cv2.imwrite("./TEST CAM.jpg", utils.stitch_images(heatmaps))


def make_bias_metrics():
    img_u = DatasetLoader("./dataset", 10000)
    img_segmented = DatasetLoader("./segmentedDB", 10000)

    for p in img_segmented.imgDataArray:
        p.name[:-4] + "_final_masked.jpg"

    dummy_model = get_custom_model("GAP", True)
    vgg16 = VGG16FineTuned(img_u)
    graph_context = tf.get_default_graph()

    bias_metric = BiasMetric(graph_context)
    mc = MetricCallback(bias_metric, dummy_model, 1, img_u, img_segmented)

    vgg16.train(1, False, [mc])

    bias_metric.save_to_csv()


def main():
    np.random.seed(123)  # for reproducibility
    make_bias_metrics()
    # img_u = DatasetLoader("./dataset", 10000)
    # vgg16 = VGG16FineTuned(img_u)
    # vgg16.train()

if __name__ == "__main__":
    main()

from vis.utils import utils

from plant_village_custom_model import *
from model_utils import get_heatmap


# https://elitedatascience.com/keras-tutorial-deep-learning-in-python#step-1
# http://cnnlocalization.csail.mit.edu/
# https://arxiv.org/pdf/1312.4400.pdf
# https://www.quora.com/What-is-global-average-pooling
# https://arxiv.org/pdf/1512.04150.pdf
# https://arxiv.org/pdf/1512.03385.pdf


def get_custom_model(mode, random_weights):
    return get_custom_model(mode, random_weights)


def create_cam():
    model = get_custom_model("custom", "GAP")
    heatmaps = []
    for path in next(os.walk('./visual'))[2]:
        # Predict the corresponding class for use in `visualize_saliency`.
        seed_img = utils.load_img('./visual/' + path, target_size=(256, 256))

        # Here we are asking it to show attention such that prob of `pred_class` is maximized.
        heatmap = get_heatmap(seed_img, model, "Conv4")
        heatmaps.append(heatmap)

    cv2.imwrite("./TEST CAM.jpg", utils.stitch_images(heatmaps))


def main():
    np.random.seed(123)  # for reproducibility
    vgg16 = get_custom_model(None, None)


if __name__ == "__main__":
    main()

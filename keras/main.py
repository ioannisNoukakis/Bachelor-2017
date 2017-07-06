import json
import os
import sys

import tensorflow as tf
from PIL import Image

from VGG16_ft import VGG16FineTuned
from heatmapgenerate import heatmap_generate
from plant_village_custom_model import *

# https://elitedatascience.com/keras-tutorial-deep-learning-in-python#step-1
# http://cnnlocalization.csail.mit.edu/
# https://arxiv.org/pdf/1312.4400.pdf
# https://www.quora.com/What-is-global-average-pooling
# https://arxiv.org/pdf/1512.04150.pdf
# https://arxiv.org/pdf/1512.03385.pdf

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


def main():
    np.random.seed(123)  # for reproducibility

    argv = sys.argv
    if argv[1] == "0":
        vggft = VGG16FineTuned(dataset_loader=DatasetLoader(argv[2], 10000), mode=argv[4])
        vggft.train(int(argv[5]), weights_out=argv[3])
    if argv[1] == "1":
        dl = DatasetLoader(argv[2], 10000)
        vggft = VGG16FineTuned(dataset_loader=dl, mode=argv[4])
        vggft.train(int(argv[5]), weights_in=argv[3])

        # plot CAMs only for the validation data:
        for i in range(dl.number_of_imgs_for_train, dl.number_of_imgs):
            outpath = "maps/" + dl.imgDataArray[i].directory + "/" + dl.imgDataArray[i].name
            for j in range(0, dl.nb_classes):
                outname = outpath + "/" + str(j) + ".png"

                try:
                    os.makedirs("maps/" + dl.imgDataArray[i].directory + "/" + dl.imgDataArray[i].name)
                except OSError:
                    pass

                input_img = Image.open(
                    dl.baseDirectory + "/" + dl.imgDataArray[i].directory + "/" + dl.imgDataArray[i].name)
                heatmap = heatmap_generate(
                    graph_context=tf.get_default_graph(),
                    input_img=input_img,
                    model=vggft.model,
                    class_to_predict=j,
                    layer_name='CAM')
                heatmap.save(outname)
            predict_input = cv2.imread(dl.baseDirectory + "/" + dl.imgDataArray[i].directory + "/" +
                                       dl.imgDataArray[i].name, cv2.IMREAD_COLOR)
            predict_input = np.expand_dims(predict_input, axis=1)
            results = vggft.model.predict(predict_input)
            with open(outpath + '/resuts.json', 'w') as outfile:
                json.dump({'predicted': 'nan', "true_label": dl.imgDataArray[i].img_class}, outfile)


if __name__ == "__main__":
    main()

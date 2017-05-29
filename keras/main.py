from concurrent.futures import ThreadPoolExecutor

from quiver_engine.layer_result_generators import get_outputs_generator
# from quiver_engine.server import get_evaluation_context_getter
from quiver_engine.util import *
from sklearn.preprocessing import MinMaxScaler
from skimage.measure import compare_ssim as ssim

from imgUtils import *

from PIL import Image, ImageEnhance, ImageOps

from vis.utils.vggnet import VGG16
import scipy.misc

from keras.preprocessing.image import img_to_array
from vis.utils import utils
from vis.visualization import visualize_cam

from plant_village_custom_model import *


# https://elitedatascience.com/keras-tutorial-deep-learning-in-python#step-1
# http://cnnlocalization.csail.mit.edu/
# https://arxiv.org/pdf/1312.4400.pdf
# https://www.quora.com/What-is-global-average-pooling
# https://arxiv.org/pdf/1512.04150.pdf
# https://arxiv.org/pdf/1512.03385.pdf


def get_model(name, mode):
    if name == "custom":
        return get_custom_model(mode)
    elif name == "VGG16":
        return VGG16(weights='imagenet', include_top=True)


def reduce_opacity(im, opacity):
    """
    Returns an image with reduced opacity.
    Taken from http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/362879
    """
    assert 0 <= opacity <= 1
    if im.mode != 'RGBA':
        im = im.convert('RGBA')
    else:
        im = im.copy()
    alpha = im.split()[3]
    alpha = ImageEnhance.Brightness(alpha).enhance(opacity)
    im.putalpha(alpha)
    return im


def get_heatmap(input_img, model, layer_name, image_name=None):
    input_img = preprocess_input(np.expand_dims(image.img_to_array(input_img), axis=0), dim_ordering='default')
    output_generator = get_outputs_generator(model, layer_name)
    layer_outputs = output_generator(input_img)[0]
    heatmap = Image.new("RGBA", (256, 256), color=0)
    w = MinMaxScaler((0, 0.9)).fit_transform((model.get_layer("W").get_weights()[0]).flatten())

    for z in range(0, layer_outputs.shape[2]):
        img = layer_outputs[:, :, z]

        deprocessed = scipy.misc.toimage(img).resize((256, 256)).convert("RGBA")
        datas = deprocessed.getdata()
        new_data = []
        for item in datas:
            if item[0] < 16 and item[1] < 16 and item[2] < 16:
                new_data.append((0, 0, 0, 0))
            else:
                new_data.append(item)
        deprocessed.putdata(new_data)
        deprocessed = reduce_opacity(deprocessed, w[z])
        heatmap.paste(deprocessed, (0, 0), deprocessed)
    ImageOps.invert(heatmap.convert("RGB")).convert("RGBA").save("TMP.png", "PNG")
    heatmap = cv2.imread("TMP.png", cv2.CV_8UC3)

    heatmap = np.maximum(heatmap, 0)

    heatmap_colored = cv2.applyColorMap(np.uint8(255 * np.asarray(heatmap)), cv2.COLORMAP_JET)
    heatmap_colored[np.where(heatmap <= 0.2)] = 0

    if image_name is not None:
        heatmap_colored = cv2.putText(heatmap_colored, image_name, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0),
                                      2)
    return cv2.resize(heatmap_colored, (224, 224))


def get_VGG16_heatmap(model, layer_idx, seed_img):
    pred_class = np.argmax(model.predict(np.array([img_to_array(seed_img)])))

    # Here we are asking it to show attention such that prob of `pred_class` is maximized.
    return visualize_cam(model, layer_idx, [pred_class], seed_img, overlay=False)


def detect_bias(nb_threads):
    with ThreadPoolExecutor(max_workers=nb_threads) as executor:
        """Lecture : http://www.pyimagesearch.com/2014/09/15/python-compare-two-images/"""

        model_custom = get_model("custom", "GAP")

        model_vgg16 = get_model("VGG16", "-")
        layer_name = 'predictions'
        layer_idx = [idx for idx, layer in enumerate(model_vgg16.layers) if layer.name == layer_name][0]

        img_u = DatasetLoader("./dataset", 10000)
        score = 0
        j = 0

        while img_u.has_next():
            future = []
            for i in range(0, nb_threads):

                if not img_u.has_next():
                    break

                next_path = img_u.next_path()

                im = Image.open(next_path)
                heatmapCustom = get_heatmap(im, model_custom, "Conv4")

                seed_img = utils.load_img(next_path, target_size=(224, 224))
                heatmapVGG16 = get_VGG16_heatmap(model_vgg16, layer_idx, seed_img)

                future.append(executor.submit(ssim, heatmapCustom, heatmapVGG16,
                                              multichannel=True))

            for i in range(0, len(future)):
                try:
                    score += future[i].result()
                    j += 1
                    print(j, "/", img_u.get_nb_images())
                except ValueError:
                    print("Value error -> skipped")

        score = score/j
        print("THE DATASET", "dataset", "HAS A SCORE OF", score)


def create_cam():
    model = get_model("custom", "GAP")
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
    detect_bias(8)

if __name__ == "__main__":
    main()

from sklearn.preprocessing import MinMaxScaler
from img_loader import *
from PIL import Image
import scipy.misc

from model_utils import reduce_opacity, get_outputs_generator


def heatmap_generate(input_img, model, class_to_predict, layer_name, image_name=None, tmp_name = 'tmp.png'):
    """
    Generate a heatmap for the bias metrics.

    :param tmp_name:
    :param class_to_predict: The class for the heatmap to be generated
    :param graph_context: the tensorflow context.
    :param input_img: the image to generate heatmap.
    :param model: the model.
    :param layer_name: The layer name that will be used to generate the heatmap.
    :param image_name: print a text on the image.
    :return:the heatmap or None if an error occured.
    """
    output_generator = get_outputs_generator(model, layer_name)
    layer_outputs = output_generator(np.expand_dims(input_img, axis=0))[0]
    heatmap = Image.new("RGBA", (224, 224), color=0)
    # Normalize input on weights
    w = MinMaxScaler((0.0, 1.0)).fit_transform(model.get_layer("W").get_weights()[0])

    for z in range(0, layer_outputs.shape[2]):  # Iterate through the number of kernels
        img = layer_outputs[:, :, z]

        deprocessed = scipy.misc.toimage(cv2.resize(img, (224, 224))).convert("RGBA")
        deprocessed = reduce_opacity(deprocessed, w[z][class_to_predict])
        heatmap.paste(deprocessed, (0, 0), deprocessed)
    # heatmap = image.img_to_array(ImageOps.invert(heatmap.convert("RGB")).convert("RGBA"))
    # ImageOps.invert(heatmap.convert("RGB")).convert("RGBA").save("TMP.png", "PNG")
    heatmap.save(tmp_name, "PNG")
    heatmap = cv2.imread(tmp_name, cv2.CV_8UC3)  # FIXME: remove tmp file

    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

    if image_name is not None:
        heatmap_colored = cv2.putText(heatmap_colored, image_name, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0),
                                      2)
    return Image.fromarray(cv2.resize(heatmap_colored, (224, 224)))

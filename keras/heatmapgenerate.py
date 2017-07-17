from sklearn.preprocessing import MinMaxScaler
from img_loader import *
from PIL import Image
import scipy.misc
import tensorflow as tf
from keras import backend as K

from model_utils import reduce_opacity, get_outputs_generator


def cam_generate_tf_ops(session, input_img, model, class_to_predict, layer_name, im_width=256):
    w = model.get_layer("W").get_weights()[0]
    output_generator = get_outputs_generator(model, layer_name)
    layer_outputs = output_generator(np.expand_dims(input_img[0], axis=0))[0]
    layer_outputs_alt = layer_outputs.reshape(1, 8, 8, 512)
    conv_resized = tf.image.resize_bilinear(layer_outputs_alt, [im_width, im_width])
    maps = K.dot(conv_resized, tf.convert_to_tensor(w))
    maps = tf.reshape(maps, (38, 256, 256))
    with session:
        maps_arr = maps.eval()
    maps_arr[class_to_predict] = MinMaxScaler((0.0, 1.0)).fit_transform(maps_arr[class_to_predict])
    return maps_arr[class_to_predict]


# To be fully confirmed but initials tests pass so ready for production.
def cam_generate_for_vgg16(input_img, model, class_to_predict, layer_name, image_name=None, color=False):
    output_generator = get_outputs_generator(model, layer_name)
    layer_outputs = output_generator(np.expand_dims(input_img, axis=0))[0]
    w = model.get_layer("W").get_weights()[0]

    heatmap = cv2.resize(layer_outputs[:, :, 0], (224, 224))
    heatmap *= w[0][class_to_predict]

    for z in range(1, layer_outputs.shape[2]):  # Iterate through the number of kernels
        img = cv2.resize(layer_outputs[:, :, z], (224, 224))
        heatmap += img * w[z][class_to_predict]

    heatmap = MinMaxScaler((0.0, 1.0)).fit_transform(heatmap)

    if color:
        heatmap_colored = cv2.applyColorMap(np.uint8(heatmap), cv2.COLORMAP_JET)

        if image_name is not None:
            heatmap_colored = cv2.putText(heatmap_colored, image_name, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                          (0, 0, 0),
                                          2)
        return heatmap_colored
    else:
        return heatmap



"""
def heatmap_generate(input_img, model, class_to_predict, layer_name, image_name=None, tmp_name='tmp.png'):

    Generate a heatmap for the bias metrics.

    :param tmp_name:
    :param class_to_predict: The class for the heatmap to be generated
    :param graph_context: the tensorflow context.
    :param input_img: the image to generate heatmap.
    :param model: the model.
    :param layer_name: The layer name that will be used to generate the heatmap.
    :param image_name: print a text on the image.
    :return:the heatmap or None if an error occured.

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
    heatmap.save('tmp/' + tmp_name, "PNG")
    heatmap = cv2.imread('tmp/' + tmp_name, cv2.CV_8UC3)  # FIXME: remove tmp file

    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

    if image_name is not None:
        heatmap_colored = cv2.putText(heatmap_colored, image_name, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0),
                                      2)
    return Image.fromarray(cv2.resize(heatmap_colored, (224, 224)))
"""
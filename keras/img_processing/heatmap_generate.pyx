from quiver_engine.util import *
from sklearn.preprocessing import MinMaxScaler
from img_loader import *
from PIL import Image, ImageOps
import scipy.misc

from model_utils import reduce_opacity, get_outputs_generator

def heatmap_generate(graph_context, input_img, model, layer_name, image_name=None):
    """
    Generate a heatmap for the bias metrics.

    :param graph_context: the tensorflow context.
    :param input_img: the image to generate heatmap.
    :param model: the model.
    :param layer_name: The layer name that will be used to generate the heatmap.
    :param image_name: print a text on the image.
    :return:the heatmap or None if an error occured.
    """
    try:
        with graph_context.as_default():
            input_img = preprocess_input(np.expand_dims(image.img_to_array(input_img), axis=0), dim_ordering='default')
            output_generator = get_outputs_generator(model, layer_name)
            layer_outputs = output_generator(input_img)[0]
            heatmap = Image.new("RGBA", (224, 224), color=0)
            # Normalize input on weights
            w = MinMaxScaler((0.0, 1.0)).fit_transform((model.get_layer("W").get_weights()).flatten())

            for z in range(0, layer_outputs.shape[2]):
                img = layer_outputs[:, :, z]

                deprocessed = scipy.misc.toimage(img).resize((224, 224)).convert("RGBA")
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
            heatmap = cv2.imread("TMP.png", cv2.CV_8UC3)  # FIXME: remove tmp file

            heatmap_colored = cv2.applyColorMap(np.uint8(255 * np.asarray(heatmap)), cv2.COLORMAP_JET)
            heatmap_colored[np.where(heatmap <= 0.2)] = 0

            if image_name is not None:
                heatmap_colored = cv2.putText(heatmap_colored, image_name, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0),
                                              2)
            return Image.fromarray(cv2.resize(heatmap_colored, (224, 224)))
    except AssertionError:
        return None


def get_heatmap(input_img, model, layer_name, image_name=None, cv=False):
    """see above"""
    input_img = preprocess_input(np.expand_dims(image.img_to_array(input_img), axis=0), dim_ordering='default')
    output_generator = get_outputs_generator(model, layer_name)
    layer_outputs = output_generator(input_img)[0]
    heatmap = Image.new("RGBA", (224, 224), color=0)
    # Normalize input on weights
    w = MinMaxScaler((0.0, 1.0)).fit_transform((model.get_layer("W").get_weights()[0]).flatten())

    for z in range(0, layer_outputs.shape[2]):
        img = layer_outputs[:, :, z]

        deprocessed = scipy.misc.toimage(img).resize((224, 224)).convert("RGBA")
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
    heatmap = cv2.imread("TMP.png", cv2.CV_8UC3)  # FIXME: remove tmp file

    heatmap_colored = cv2.applyColorMap(np.uint8(255 * np.asarray(heatmap)), cv2.COLORMAP_JET)
    heatmap_colored[np.where(heatmap <= 0.2)] = 0

    if image_name is not None:
        heatmap_colored = cv2.putText(heatmap_colored, image_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 2, 0)

    if cv:
        return cv2.resize(heatmap_colored, (224, 224))
    else:
        return Image.fromarray(cv2.resize(heatmap_colored, (224, 224)))




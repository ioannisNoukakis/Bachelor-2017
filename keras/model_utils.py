from quiver_engine.layer_result_generators import get_outputs_generator
from quiver_engine.util import *
from sklearn.preprocessing import MinMaxScaler
from img_loader import *
from PIL import Image, ImageEnhance, ImageOps
import scipy.misc


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


# FIXME: Better check the way that vis create the output generator
def get_heatmap(input_img, model, layer_name, image_name=None):
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
    heatmap = cv2.imread("TMP.png", cv2.CV_8UC3) # FIXME: remove tmp file

    heatmap_colored = cv2.applyColorMap(np.uint8(255 * np.asarray(heatmap)), cv2.COLORMAP_JET)
    heatmap_colored[np.where(heatmap <= 0.2)] = 0

    if image_name is not None:
        heatmap_colored = cv2.putText(heatmap_colored, image_name, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0),
                                      2)
    return cv2.resize(heatmap_colored, (224, 224))


def compareCAMSTwoModels(input_img, mask_img, vgg16_ft, custom_r):
    cam_a = get_heatmap(input_img, vgg16_ft, 'block5_conv3')
    cam_b = get_heatmap(input_img, custom_r, 'Conv4')


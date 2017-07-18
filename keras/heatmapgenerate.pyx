import time

import json
from PIL import Image
from tensorflow.python.ops.image_ops_impl import ResizeMethod

from img_loader import *
import tensorflow as tf
import numpy as np
from keras import backend as K

from model_utils import get_outputs_generator
from keras.applications.imagenet_utils import preprocess_input


def generate_maps(dl: DatasetLoader, model, map_out: str, all_classes=True, batch_size=10, mode='cv2'):
    with K.get_session():
        o_generator = get_outputs_generator(model, 'CAM')
        o_resizer = tf.image.resize_images
        o_dot = K.dot
        # plot CAMs only for the validation data:
        k = 0
        img_arr = []
        for i in range(dl.number_of_imgs_for_train, dl.number_of_imgs):
            if i == dl.number_of_imgs-1:
                k = batch_size-1
            rpath = dl.baseDirectory + "/" + dl.imgDataArray[i].directory + "/" + dl.imgDataArray[i].name
            img = cv2.imread(rpath, cv2.IMREAD_COLOR)
            # print('!!!!!!!!debug', rpath, i)
            img_arr.append(img)
            k += 1
            if k == batch_size:
                start_time = time.time()
                predict_input = np.asarray(img_arr,  dtype='float32')
                predict_input = preprocess_input(predict_input)

                layer_outputs = o_generator(predict_input)

                predictions = model.predict(predict_input)

                if mode == 'cv2': # model, layer_outputs, nb_classes, im_width=256):
                    maps_arr = cam_generate_cv2(model, layer_outputs, dl.nb_classes)
                else:
                    maps_arr = cam_generate_tf_ops(model, layer_outputs, o_resizer, o_dot)

                for l, prediction in enumerate(predictions):
                    inc = i-batch_size+l+1
                    outpath = map_out + "/" + dl.imgDataArray[inc].directory + "/" + dl.imgDataArray[inc].name
                    # print('[DEBUG]', outpath, inc, i, batch_size, l)

                    try:
                        os.makedirs(outpath)
                    except OSError:
                        continue

                    value = np.argmax(prediction)
                    if all_classes:
                        a = 0
                        b = dl.nb_classes
                    else:
                        a = value
                        b = value + 1
                    for j in range(a, b):
                        outname = outpath + "/" + str(j) + '.tiff'
                        if mode == 'cv2':
                            Image.fromarray(maps_arr[l][j]).save(outname)
                        else:
                            Image.fromarray(maps_arr[l, :, :, j]).save(outname)
                        with open(outpath + '/resuts.json', 'w') as outfile:
                            json.dump({'predicted': str(value), "true_label": str(dl.imgDataArray[i].img_class)},
                                      outfile)
                print("got cams in", time.time() - start_time)
                k = 0
                img_arr = []


def cam_generate_tf_ops(model, layer_outputs, resizer, dot, im_width=256):
    w = model.get_layer("W").get_weights()[0]

    conv_resized = resizer(layer_outputs, [im_width, im_width], method=ResizeMethod.BICUBIC, )
    maps = dot(conv_resized, tf.convert_to_tensor(w))
    maps_arr = maps.eval()
    return maps_arr


def cam_generate_cv2(model, layer_outputs, nb_classes, im_width=256):
    w = model.get_layer("W").get_weights()[0]
    maps_arr = []
    for i in range(0, layer_outputs.shape[0]):
        heatmap_arr = []
        for j in range(0, nb_classes):
            heatmap = cv2.resize(layer_outputs[i, :, :, 0], (im_width, im_width), interpolation=cv2.INTER_CUBIC)
            heatmap *= w[0][j]
            for z in range(1, layer_outputs.shape[3]):  # Iterate through the number of kernels
                img = cv2.resize(layer_outputs[i, :, :, z], (im_width, im_width), interpolation=cv2.INTER_CUBIC)
                heatmap += img * w[z][j]

            heatmap_arr.append(heatmap)
        maps_arr.append(heatmap_arr)
    return np.asarray(maps_arr, dtype='float32')


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
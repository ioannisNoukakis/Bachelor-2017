import time

import json
from PIL import Image, ImageOps
from tensorflow.python.ops.image_ops_impl import ResizeMethod

from img_loader import *
import tensorflow as tf
import numpy as np
from keras import backend as K
from scipy.misc import toimage
from keras.applications.imagenet_utils import preprocess_input

from img_processing import reduce_opacity
from model_utils import get_outputs_generator


def generate_maps(dl: DatasetLoader, model, map_out: str, graph, all_classes=True, batch_size=10, mode='cv2'):
    """
    Generates class activation mappings for the test part of a data set. For example, if you have 38 classes and 13000
    samples on your test data set this will generate 38*13000 new images. It is recommended to have at least 500 GB free
    disk space to perform this operation.

    :param dl: The dataset loader (input)
    :param model: The model that will be used to draw CAMs.
    :param map_out: the out folder to save the generated heatmaps (500BG free recommended).
    :param graph: The current Tensorflow graph. In any doubt, just call tf.get_default_graph() for this parameter.
    :param all_classes: if true, will generate CAMs for every classes for every sample. If false will only generate the
    CAM for the predicted class.
    :param batch_size: umber of images to load into the GPU (if any). If on GPU recommended to load only GPU memory * 2 + 1
    :param mode: Here you can choose witch implementation of the CAM algorithm to use. If cv2 will use opencv. Else will use
    Tensorflow.
    :return: -
    """
    # o_generator = get_outputs_generator(model, 'CAM')
    input = model.input
    output = model.get_layer('CAM').output
    output_predict = model.get_layer('W').output
    output_fn = K.function([input], [output])
    fn_predict = K.function([input], [output_predict])

    o_resizer = tf.image.resize_images
    o_dot = K.dot

    # plot CAMs only for the validation data:
    k = 0
    counter = 0
    img_arr = []
    with K.get_session() as sess:
        in_place = tf.placeholder(tf.float32, [None, None, None, 512])
        size_place = tf.placeholder(tf.int32, [2])
        convert_place = tf.placeholder(tf.float32, [512, len(dl.directories)])
        first_func = o_resizer(in_place, size_place, ResizeMethod.BICUBIC)
        second_func = o_dot(in_place, convert_place)
        graph.finalize()

        for i in range(dl.number_of_imgs_for_train, dl.number_of_imgs):
            with graph.as_default() as gr:
                if i == dl.number_of_imgs - 1:
                    k = batch_size - 1
                rpath = dl.baseDirectory + "/" + dl.imgDataArray[i].directory + "/" + dl.imgDataArray[i].name
                img = cv2.imread(rpath, cv2.IMREAD_COLOR)
                # print('!!!!!!!!debug', rpath, i)
                img_arr.append(img)
                k += 1
                if k == batch_size:
                    start_time = time.time()
                    predict_input = np.asarray(img_arr, dtype='float32')
                    predict_input = preprocess_input(predict_input)

                    k = 0
                    img_arr = []

                    layer_outputs = output_fn([predict_input])[0]
                    predictions = fn_predict([predict_input])[0]

                    if mode == 'cv2':  # model, layer_outputs, nb_classes, im_width=256):
                        maps_arr = cam_generate_cv2(model, layer_outputs, dl.nb_classes)
                    else:
                        maps_arr = cam_generate_tf_ops(model, layer_outputs, sess, first_func, second_func, in_place,
                                                       size_place,
                                                       convert_place)

                    for l, prediction in enumerate(predictions):
                        inc = i - batch_size + l + 1
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
                                json.dump({'predicted': str(value), "true_label": str(dl.imgDataArray[inc].img_class)},
                                          outfile)
                        print("cam(", counter, "/", dl.number_of_imgs_for_test, "completed")
                        counter += 1
                    print("got cams in", time.time() - start_time)


def cam_generate_tf_ops(model, layer_outputs, sess, first_func, second_func, in_place, size_place, convert_place,
                        im_width=256):
    """
    Generates heatmaps in the Tensorflow computation graph.
    :param model: the model.
    :param layer_outputs: the activated kernels of the model.
    :param sess: the Tensorflow session.
    :param first_func: tf.image.resize_images
    :param second_func: K.dot
    :param in_place: Tensorflow placeholder for layer_outputs
    :param size_place: Tensorflow placeholder for resize parameter.
    :param convert_place: Tensorflow placeholder for GAP weights.
    :param im_width: size of the image (size to resize).
    :return: the generated heatmaps.
    """
    conv_resized = sess.run(first_func, feed_dict={in_place: layer_outputs, size_place: [im_width, im_width]})

    w = model.get_layer("W").get_weights()[0]
    maps = sess.run(second_func, feed_dict={in_place: conv_resized, convert_place: w})
    return maps


def cam_generate_cv2(model, layer_outputs, nb_classes, im_width=256):
    """
    Generates heatmaps.
    :param model: the model.
    :param layer_outputs: the activated kernels of the model.
    :param nb_classes: the number of classes in the model.
    :param im_width: size of the image (size to resize).
    :return: the generated heatmaps.
    """
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


def create_cam_colored(dl: DatasetLoader, model, outname: str, im_width=256, n=8, s=256):
    """
    Creates a nice colored printing of class activation mappings for every first sample of dataset and a model. Generates
    one image out of all the computed CAMs.

    :param dl: the dataset
    :param model: the model
    :param outname: the path for saving the result.
    :param im_width: size of the image (size to resize).
    :param n: n columns
    :param s: size of the image.
    :return: -
    """

    heatmaps = []
    for i in range(0, dl.nb_classes):
        predict_input = (cv2.imread(dl.baseDirectory + "/" + dl.picker[i].directory + "/" +
                                    dl.picker[i].name, cv2.IMREAD_COLOR))
        base = Image.open(dl.baseDirectory + "/" + dl.picker[i].directory + "/" +
                          dl.picker[i].name)
        predict_input = predict_input.astype('float32')
        predict_input = np.expand_dims(predict_input, axis=0)
        predict_input = preprocess_input(predict_input)

        output_generator = get_outputs_generator(model, 'CAM')
        layer_outputs = output_generator(predict_input)[0]

        inputs = model.input
        output_predict = model.get_layer('W').output
        fn_predict = K.function([inputs], [output_predict])
        prediction = fn_predict([predict_input])[0]
        value = np.argmax(prediction)

        w = model.get_layer("W").get_weights()[0]
        heatmap = cv2.resize(layer_outputs[:, :, 0], (im_width, im_width), interpolation=cv2.INTER_CUBIC)
        heatmap *= w[0][value]
        for z in range(1, layer_outputs.shape[2]):  # Iterate through the number of kernels
            img = cv2.resize(layer_outputs[:, :, z], (im_width, im_width), interpolation=cv2.INTER_CUBIC)
            heatmap += img * w[z][value]

        heatmap = cv2.applyColorMap(np.uint8(np.asarray(ImageOps.invert(toimage(heatmap)))), cv2.COLORMAP_JET)
        heatmap = cv2.putText(heatmap, str(dl.picker[i].img_class), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0),
                              2)
        heatmap = toimage(heatmap)
        heatmap = reduce_opacity(heatmap, 0.5)
        base.paste(heatmap, (0, 0), heatmap)
        heatmaps.append(base)

    result = Image.new("RGB", (n * s, (len(heatmaps) // n + 1) * s))
    for index, img in enumerate(heatmaps):
        x = index % n * 256
        y = index // n * 256
        w, h = img.size
        print('pos {0},{1} size {2},{3}'.format(x, y, w, h))
        result.paste(img, (x, y, x + w, y + h))

    result.save(outname)
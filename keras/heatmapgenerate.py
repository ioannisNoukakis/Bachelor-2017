from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.ops.image_ops_impl import ResizeMethod

from img_loader import *
import tensorflow as tf
from keras import backend as K


def cam_generate_tf_ops(input_img, model, class_to_predict, output_generator, im_width=256):
    w = model.get_layer("W").get_weights()[0]
    layer_outputs = output_generator(np.expand_dims(input_img, axis=0))[0]
    conv_resized = tf.image.resize_images(layer_outputs, [im_width, im_width], method=ResizeMethod.BICUBIC, )
    maps = K.dot(conv_resized, tf.convert_to_tensor(w))
    maps_arr = maps.eval()
    return maps_arr[:, :, class_to_predict]


# To be fully confirmed but initials tests pass so ready for production.
def cam_generate_cv2(input_img, model, class_to_predict, output_generator, im_width=256):
    layer_outputs = output_generator(np.expand_dims(input_img, axis=0))[0]
    w = model.get_layer("W").get_weights()[0]

    heatmap = cv2.resize(layer_outputs[:, :, 0], (im_width, im_width), interpolation=cv2.INTER_CUBIC)
    heatmap *= w[0][class_to_predict]

    for z in range(1, layer_outputs.shape[2]):  # Iterate through the number of kernels
        img = cv2.resize(layer_outputs[:, :, z], (im_width, im_width), interpolation=cv2.INTER_CUBIC)
        heatmap += img * w[z][class_to_predict]

    return heatmap
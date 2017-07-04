import keras.backend as K
from keras import applications
from keras.models import Sequential
from keras.layers.core import Flatten, Dense
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D
from keras.layers.pooling import AveragePooling2D
from keras.layers.convolutional import ZeroPadding2D
import matplotlib.pylab as plt
import numpy as np
import theano.tensor.nnet.abstract_conv as absconv
import cv2
import h5py
import os
import tensorflow as tf

from keras.utils import np_utils

from img_loader import DatasetLoader


def VGGCAM(nb_classes, num_input_channels=1024):
    """
    Build Convolution Neural Network

    args : nb_classes (int) number of classes

    returns : model (keras NN) the Neural Net model
    """

    model = Sequential(applications.VGG16(weights='imagenet', include_top=False).layers)

    # Add another conv layer with ReLU + GAP
    model.add(Convolution2D(num_input_channels, 3, 3, activation='relu', border_mode="same", name="CAM"))
    model.add(GlobalAveragePooling2D())
    # Add the W layer
    model.add(Dense(nb_classes, activation='softmax'))

    model.name = "VGGCAM"

    return model


def get_classmap(model, X, nb_classes, batch_size, num_input_channels, ratio):
    with tf.Session():
        inc = model.layers[0].input
        conv6 = model.get_layer('CAM').output
        conv6_resized = absconv.bilinear_upsampling(conv6, ratio, batch_size=batch_size, num_input_channels=num_input_channels)

        WT = model.layers[-1].kernel
        conv6_resized = K.reshape(conv6_resized, (-1, num_input_channels, 224 * 224))
        classmap = K.dot(WT, conv6_resized).reshape((-1, nb_classes, 224, 224))
        get_cmap = K.function([inc], classmap)
        return get_cmap([X])


def train_VGGCAM(dataset_loader: DatasetLoader, n_epochs, num_input_channels=1024):
    """
    Train VGGCAM model

    args: VGG_weight_path (str) path to keras vgg16 weights
          nb_classes (int) number of classes
          num_input_channels (int) number of conv filters to add
                                   in before the GAP layer

    """
    # Load model
    model = VGGCAM(dataset_loader.nb_classes)
    print('Model loaded.')

    # Compile
    model.compile(optimizer="sgd", loss='categorical_crossentropy')

    redo = True
    for k in range(0, n_epochs):
        print("epoch", k, "/", n_epochs)
        while redo:
            redo, X, y = dataset_loader.load_dataset()
            y = np_utils.to_categorical(y, dataset_loader.nb_classes)
            for i, _ in enumerate(X):
                # pre processing
                # X = cv2.resize(X, (224, 224)).astype(np.float32)
                X = X.astype(np.float32)

                X[i][:, :, 0] -= 103.939
                X[i][:, :, 1] -= 116.779
                X[i][:, :, 2] -= 123.68
                X[i] = np.expand_dims(X[i], axis=0)

            model.fit(X, y, nb_epoch=1, verbose=1)

    # Save model
    model.save_weights(os.path.join('FT_VGG16_weights.h5'))


def plot_classmap(outname, img_path, label,
                  nb_classes,
                  VGGCAM_weight_path="./FT_VGG16_weights.h5",
                  num_input_channels=1024, ratio=16):
    """
    Plot class activation map of trained VGGCAM model

    args: VGGCAM_weight_path (str) path to trained keras VGGCAM weights
          img_path (str) path to the image for which we get the activation map
          label (int) label (0 to nb_classes-1) of the class activation map to plot
          nb_classes (int) number of classes
          num_input_channels (int) number of conv filters to add
                                   in before the GAP layer
          ratio (int) upsampling ratio (16 * 14 = 224)

    """

    # Load and compile model
    model = VGGCAM(nb_classes, num_input_channels)
    model.load_weights(VGGCAM_weight_path)
    model.compile(loss="categorical_crossentropy", optimizer="sgd")

    # Load and format data
    im = cv2.resize(cv2.imread(img_path), (224, 224)).astype(np.float32)
    # VGG model normalisations
    im[:, :, 0] -= 103.939
    im[:, :, 1] -= 116.779
    im[:, :, 2] -= 123.68
    im = im.transpose((2, 0, 1))
    im = im.reshape(1, 3, 224, 224)

    batch_size = 1
    classmap = get_classmap(model,
                            im,
                            nb_classes,
                            batch_size,
                            num_input_channels=num_input_channels,
                            ratio=ratio)

    plt.imsave(outname, classmap[0, label, :, :], cmap="jet")
    return model.predict(im)

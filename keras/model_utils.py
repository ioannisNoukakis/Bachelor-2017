from keras.utils import np_utils
from img_loader import *
from PIL import ImageEnhance
from keras.models import Model

from logger import info


def train_model(model, dataset_loader: DatasetLoader, n_epochs, callbacks):
    """
    Trains a model. At the end of each epochs evaluates it.
    :param model: The model to be trained
    :param dataset_loader: The data set loader with the model will train.
    :param n_epochs: The number of iterations over the data.
    :param callbacks: keras callbacks
    :return: The trained model and its score
    """
    redo = True
    score = []
    for i in range(0, n_epochs):
        info("[MODEL-UTILS] epoch", i)
        while redo:
            redo, x_train, y_train = dataset_loader.load_dataset()
            # Preprocessing
            x_train = x_train.astype('float32')
            x_train /= 255

            y_train = np_utils.to_categorical(y_train, dataset_loader.nb_classes)

            # Fit model on training data
            info("[MODEL-UTILS] Starting...", "")
            if callbacks:
                model.fit(x_train, y_train, batch_size=10, nb_epoch=1, verbose=0, callbacks=callbacks)
            else:
                model.fit(x_train, y_train, batch_size=10, nb_epoch=1, verbose=1)
            # TODO: Maybe this is the wrong order of how to apply epochs -> investigate
            score = evaluate_model(model, dataset_loader, score)
        redo = True
    return model, score


def evaluate_model(model, dataset_loader: DatasetLoader, score):
    """
    Evaluates a model.

    :param model: The model to be evaluated.
    :param dataset_loader: The data set loader that will provide the evaluation data.
    :param score: The model's score
    :return: the new score
    """
    redo = True
    while redo:
        redo, x_test, y_test = dataset_loader.load_dataset()
        # Preprocessing
        x_test = x_test.astype('float32')
        x_test /= 255

        y_test_2 = np_utils.to_categorical(y_test, dataset_loader.nb_classes)
        # Evaluate model on test data
        score.append(model.evaluate(x_test, y_test_2, batch_size=10, verbose=1))
    return score


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


def get_outputs_generator(model, layer_name):
    """
    Gets the output generator of a specific layer of the model.

    :param model: The model
    :param layer_name: The layer's name
    :return: the output generator (a function)
    """
    layer_model = Model(
        input=model.input,
        output=model.get_layer(layer_name).get_output_at(0)
    )

    return layer_model.predict

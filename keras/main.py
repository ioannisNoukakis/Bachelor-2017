from keras.models import Sequential
from keras.layers import *
from keras.utils import np_utils
from quiver_engine.layer_result_generators import get_outputs_generator
from quiver_engine.server import get_evaluation_context_getter
from quiver_engine.util import *
from sklearn.preprocessing import MinMaxScaler

from imgUtils import *
from quiver_engine import server
import time
from PIL import Image, ImageEnhance, ImageOps
from pathlib import Path
from vis.utils.vggnet import VGG16
import keras
import scipy.misc

from keras.preprocessing.image import img_to_array
from vis.utils import utils
from vis.visualization import visualize_cam
from keras.models import model_from_json

import pandas as pd


# https://elitedatascience.com/keras-tutorial-deep-learning-in-python#step-1
# http://cnnlocalization.csail.mit.edu/
# https://arxiv.org/pdf/1312.4400.pdf
# https://www.quora.com/What-is-global-average-pooling
# https://arxiv.org/pdf/1512.04150.pdf


def get_model(name, mode):
    if name == "custom":
        img_u = ImgUtils("./dataset", 10000)
        start = time.strftime("%c")
        the_true_score = []
        nb_classes = img_u.discover_and_make_order()
        N_EPOCHS = 5

        # Define model architecture
        model = Sequential()

        model.add(Convolution2D(nb_filter=64, nb_row=3, nb_col=3, activation='relu', input_shape=(256, 256, 3),
                                dim_ordering='tf', name="Conv1"))
        model.add(Convolution2D(nb_filter=32, nb_row=3, nb_col=3, activation='relu', name="Conv2"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Convolution2D(nb_filter=32, nb_row=3, nb_col=3, activation='relu', name="Conv3"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Convolution2D(nb_filter=32, nb_row=3, nb_col=3, activation='relu', name="Conv4"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        if mode == "GAP":
            model.add(GlobalAveragePooling2D(name="GAP"))
        elif mode == "dense":
            model.add(Flatten())
            model.add(Dense(128, activation='relu'))
            model.add(Dropout(0.5))
        model.add(Dense(nb_classes, activation='softmax', name='W'))

        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        w_file = Path("./model.h5")
        model.summary()

        if w_file.is_file():
            model.load_weights("./model.h5")
            return model
        else:
            # Train
            redo = True
            # for i in range(0, N_EPOCHS):
            while redo:
                redo, x_train, y_train = img_u.load_dataset()
                # Preprocessing
                x_train = x_train.astype('float32')
                x_train /= 255

                y_train = np_utils.to_categorical(y_train, nb_classes)

                # Fit model on training data
                print("Starting...")
                model.fit(x_train, y_train, batch_size=10, nb_epoch=N_EPOCHS, verbose=1)
                # TODO: Maybe this is the wrong order of how to apply epochs -> investigate
                # redo = True

            print("Train completed! Will now evalutate...")

            # Evaluate
            redo = True
            while redo:
                redo, x_test, y_test = img_u.load_dataset()
                # Preprocessing
                x_test = x_test.astype('float32')
                x_test /= 255

                y_test_2 = np_utils.to_categorical(y_test, nb_classes)
                # Evaluate model on test data
                the_true_score.append(model.evaluate(x_test, y_test_2, batch_size=10, verbose=1))

                # y_hat = model.predict_classes(x_test)
                # print(pd.crosstab(y_hat, y_test))

            print("Model:")
            model.summary()
            print("Obtained the score:", the_true_score)
            print("Training started at:", start)
            print("Training ended at:", time.strftime("%c"))
            print("Classes:", nb_classes)
            print("Nb_epoch:", 10)
            model.save_weights("./model.h5")
            return model
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
    ImageOps.invert(heatmap.convert("RGB")).save("TMP.jpg", "JPEG")
    heatmap = cv2.imread("TMP.jpg", cv2.CV_8UC3)
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * np.asarray(heatmap)), cv2.COLORMAP_JET)
    heatmap_colored[np.where(heatmap <= 0.2)] = 0
    if image_name is not None:
        heatmap_colored = cv2.putText(heatmap_colored, image_name, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0),
                                      2)
    return heatmap_colored


def main():
    np.random.seed(123)  # for reproducibility

    model = get_model("custom", "GAP")
    heatmaps = []
    for path in next(os.walk('./visual'))[2]:
        im = Image.open('./visual/' + path)
        im = preprocess_input(np.expand_dims(image.img_to_array(im), axis=0), dim_ordering='default')
        heatmap = get_heatmap(im, model, "Conv4", path)

        heatmaps.append(heatmap)

    cv2.imwrite("./grad-CAM CUSTOM.jpg", utils.stitch_images(heatmaps))
    # server.launch(model, temp_folder='./tmp', input_folder='./visual',  port=5000)
if __name__ == "__main__":
    main()

from keras.models import Sequential
from keras.layers import *
from keras.utils import np_utils
from imgUtils import *
from quiver_engine import server
import time
from PIL import Image
from pathlib import Path
from vis.utils.vggnet import VGG16

from keras.preprocessing.image import img_to_array
from vis.utils import utils
from vis.visualization import visualize_cam
from keras.models import model_from_json

import pandas as pd
# https://elitedatascience.com/keras-tutorial-deep-learning-in-python#step-1
# http://cnnlocalization.csail.mit.edu/


def get_model(name, mode):
    if name == "custom":
        imgU = ImgUtils("./datasetNoBiais", 10000)
        start = time.strftime("%c")
        theTrueScore = []
        nb_classes = imgU.discover_and_make_order()
        N_EPOCHS = 1

        # Define model architecture
        model = Sequential()

        model.add(Convolution2D(nb_filter=64, nb_row=3, nb_col=3, activation='relu', input_shape=(256, 256, 3),
                                dim_ordering='tf'))
        model.add(Convolution2D(nb_filter=32, nb_row=3, nb_col=3, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Convolution2D(nb_filter=32, nb_row=3, nb_col=3, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Convolution2D(nb_filter=32, nb_row=3, nb_col=3, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        if mode == "avg":
            model.add(GlobalAveragePooling2D())
        else:
            model.add(Flatten())
            model.add(Dense(128, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(nb_classes, activation='softmax', name='predictions'))

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
                redo, x_train, y_train = imgU.load_dataset()
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
                redo, x_test, y_test = imgU.load_dataset()
                # Preprocessing
                x_test = x_test.astype('float32')
                x_test /= 255

                y_test_2 = np_utils.to_categorical(y_test, nb_classes)
                # Evaluate model on test data
                theTrueScore.append(model.evaluate(x_test, y_test_2, batch_size=10, verbose=1))

                # y_hat = model.predict_classes(x_test)
                # print(pd.crosstab(y_hat, y_test))

            print("Model:")
            model.summary()
            print("Obtained the score:", theTrueScore)
            print("Training started at:", start)
            print("Training ended at:", time.strftime("%c"))
            print("Classes:", nb_classes)
            print("Nb_epoch:", 10)
            model.save_weights("./model.h5")
            return model
    elif name == "VGG16":
        return VGG16(weights='imagenet', include_top=True)


def main():

    np.random.seed(123)  # for reproducibility

    model = get_model("custom", "avg")
    im = cv2.imread("./dataset/Apple___Apple_scab/0a769a71-052a-4f19-a4d8-b0f0cb75541c___FREC_Scab 3165.JPG", cv2.IMREAD_COLOR)
    im = np.expand_dims(im, axis=0)
    out = model.predict(im)

    Image.fromarray((out * 255).astype(np.uint8)).save('out.bmp')
    print(out)

    # server.launch(model, temp_folder='./tmp', input_folder='./visual',  port=5000)

if __name__ == "__main__":
    main()


from keras import applications
from keras.layers import GlobalAveragePooling2D, Dense, Convolution2D, Flatten
from keras.models import Sequential

from logger import info
from plant_village_custom_model import train_model, DatasetLoader


class VGG16FineTuned:
    """
    VGG16 fine tuned with a global average pooling layer instead of the traditional
    fully connected layer.
    """

    def __init__(self, dataset_loader: DatasetLoader, mode: str):
        """
        Create and compile the custom VGG16 model.

        :param dataset_loader: The data set loader with the model will train.
        """
        self.img_u = dataset_loader
        if mode == 'GAP_CAM':
            self.model = Sequential(applications.VGG16(weights='imagenet', include_top=False).layers)

            self.model.add(Convolution2D(512, 3, 3, activation='relu', border_mode="same", name="CAM"))
            self.model.add(GlobalAveragePooling2D(name="GAP"))
            self.model.add(Dense(dataset_loader.nb_classes, activation='softmax', name='W'))
        if mode == 'GAP':
            self.model = Sequential(applications.VGG16(weights='imagenet', include_top=False).layers)

            self.model.add(GlobalAveragePooling2D(name="GAP"))
            self.model.add(Dense(dataset_loader.nb_classes, activation='softmax', name='W'))
        if mode == 'DENSE':
            self.model = Sequential(applications.VGG16(weights='imagenet',
                                                       input_shape=(256, 256, 3),
                                                       include_top=False).layers)

            self.model.add(Flatten(name='flatten'))
            self.model.add(Dense(2048, activation='relu', name='fc1'))
            self.model.add(Dense(2048, activation='relu', name='fc2'))
            self.model.add(Dense(dataset_loader.nb_classes, activation='softmax', name='W'))

        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.summary()

    def train(self, nb_epochs, weights_in=None, weights_out=None, callbacks=None):
        """
        Trains the custom VGG16 model.

        :param weights_in:
        :param nb_epochs: the number of iterations over the data.
        :param weights_out: if the weights of the custom models should be saved.
        :param callbacks: keras callbacks
        :return:
        """
        if weights_in is None:
            self.model, score = train_model(self.model, self.img_u, nb_epochs, callbacks)
            info("[VGG16_FT]", score)
        else:
            self.model.load_weights(weights_in, by_name=True)

        if weights_out is not None and weights_in is None:
            self.model.save_weights(weights_out)

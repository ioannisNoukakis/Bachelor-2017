from keras import applications
from keras.layers import GlobalAveragePooling2D, Dense
from keras.models import Sequential

from plant_village_custom_model import train_model, DatasetLoader


class VGG16FineTuned:
    """
    VGG16 fine tuned with a global average pooling layer instead of the traditional
    fully connected layer.
    """

    def __init__(self, dataset_loader: DatasetLoader):
        """
        Create and compile the custom VGG16 model.

        :param dataset_loader: The data set loader with the model will train.
        """
        self.img_u = dataset_loader
        self.model = Sequential(applications.VGG16(weights='imagenet', include_top=False).layers)

        self.model.add(GlobalAveragePooling2D(name="GAP"))
        self.model.add(Dense(dataset_loader.nb_classes, activation='softmax', name='W'))

        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.summary()

    def train(self, nb_epochs, save_weights=False, callbacks=None):
        """
        Trains the custom VGG16 model.

        :param nb_epochs: the number of iterations over the data.
        :param save_weights: if the weights of the custom models should be saved.
        :param callbacks: keras callbacks
        :return:
        """
        self.model, score = train_model(self.model, self.img_u, nb_epochs, callbacks)
        print("score", score)
        if save_weights:
            self.model.save_weights("./VGG16_GAP.h5")

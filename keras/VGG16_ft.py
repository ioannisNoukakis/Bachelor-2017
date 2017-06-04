from keras import applications
from keras.layers import GlobalAveragePooling2D, Dense
from keras.models import Sequential

from plant_village_custom_model import train_model, evaluate_model


class VGG16FineTuned:

    def __init__(self, nb_classes):
        # https://gist.github.com/fchollet/7eb39b44eb9e16e59632d25fb3119975
        self.model = Sequential(applications.VGG16(weights='imagenet', include_top=False).layers)

        self.model.add(GlobalAveragePooling2D(name="GAP"))
        self.model.add(Dense(nb_classes, activation='softmax', name='W'))

        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train(self, img_u, callbacks = None):
        self.model = train_model(self.model, img_u, 5, callbacks)
        score = evaluate_model(self.model, img_u, [])
        print("score", score)
        self.model.save_weights("./VGG16_GAP.h5")
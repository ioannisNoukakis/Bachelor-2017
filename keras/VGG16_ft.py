from keras import applications
from keras.layers import GlobalAveragePooling2D, Dense
from keras.models import Sequential

from plant_village_custom_model import train_model, evaluate_model


class VGG16FineTuned:

    def __init__(self, img_u):
        # https://gist.github.com/fchollet/7eb39b44eb9e16e59632d25fb3119975
        self.img_u = img_u
        self.model = Sequential(applications.VGG16(weights='imagenet', include_top=False).layers)

        self.model.add(GlobalAveragePooling2D(name="GAP"))
        self.model.add(Dense(img_u.nb_classes, activation='softmax', name='W'))

        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train(self, save_weights=False, callbacks=None):
        self.model, score = train_model(self.model, self.img_u, 5, callbacks)
        print("score", score)
        if save_weights:
            self.model.save_weights("./VGG16_GAP.h5")

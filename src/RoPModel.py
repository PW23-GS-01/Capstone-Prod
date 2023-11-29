from keras import Sequential
from keras.layers import Dense

from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2S
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from tensorflow.keras.models import Model

from PIL import Image
import cv2
import numpy as np


class RoPModel:
    _IMG_SIZE = 384
    _CNN_MODEL = EfficientNetV2S(
            weights='imagenet',
            include_top=False,
            pooling='avg'
    )

    def __init__(self, model_h5, labels):
        self._labels = labels

        self._model = Sequential([
            Dense(1024, input_dim=1280, activation='relu'),
            Dense(1024, activation='relu'),
            Dense(len(self._labels), activation='softmax')
        ])

        self._model.load_weights(model_h5)

    def predict_image(self, img):
        img = img.resize((384, 384))
        img = image.img_to_array(img)
        x = np.expand_dims(img, 0)
        x = preprocess_input(x)

        x = RoPModel._CNN_MODEL.predict(x, verbose=0)
        probs = self._model.predict(x, verbose=0)

        pred = np.argmax(probs[0])

        label = self._labels[pred]
        prob = probs[0][pred]

        return label, prob

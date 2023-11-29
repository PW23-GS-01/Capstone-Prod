import io

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


class FundusImage:
    def __init__(self, image):
        self._image = Image.open(image).resize((512, 512))
        self._enh_image = None

    def get_original_image(self):
        return self._image

    def get_enhanced_image(self):
        if self._enh_image is None:
            data = self._image.copy()
            data = np.array(data)
            data = np.array([data])

            data = self._dataset_normalized(data)
            data = self._clahe_equalized(data)
            data = self._adjust_gamma(data, 1.2)
            data = data / 255.

            buffer = io.BytesIO()
            plt.imsave(buffer, data[0], cmap='gray')
            buffer.seek(0)

            self._enh_image = Image.open(buffer)

        return self._enh_image

    @staticmethod
    def _dataset_normalized(imgs):
        imgs_normalized = np.empty(imgs.shape)
        for i in range(imgs.shape[0]):
            imgs_normalized[i] = ((imgs[i] - np.min(imgs[i])) / (np.max(imgs[i]) - np.min(imgs[i]))) * 255
        return imgs_normalized

    @staticmethod
    def _clahe_equalized(imgs):
        # Create a CLAHE object (Arguments are optional).
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        imgs_equalized = np.empty(imgs.shape)

        for i in range(imgs.shape[0]):
            for c in range(3):
                # Convert to 8-bit for CLAHE (assuming the input images are 8-bit)
                img_8bit = imgs[i][:, :, c].astype(np.uint8)
                imgs_equalized[i][:, :, c] = clahe.apply(img_8bit)

        return imgs_equalized

    @staticmethod
    def _adjust_gamma(imgs, gamma=1.0):
        new_imgs = np.empty(imgs.shape)
        for i in range(imgs.shape[0]):
            for c in range(3):
                new_imgs[i][:, :, c] = cv2.LUT(np.array(imgs[i][:, :, c], dtype=np.uint8), FundusImage._create_gamma_lut(gamma))
        return new_imgs

    @staticmethod
    def _create_gamma_lut(gamma=1.0):
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
        return table

import cv2
import numpy as np
from PIL import Image


class FundusImage:
    def __init__(self, image):
        self._image = Image.open(image).resize((512, 512))
        self._enh_image = None

    def get_original_image(self):
        return self._image

    def get_enhanced_image(self):
        if self._enh_image is None:
            enhanced_image = self._image.copy()  # Create a copy of the original image

            # Check if the image is in RGB format
            if enhanced_image.mode == 'RGB':
                enhanced_image = self._clahe_equalized(enhanced_image)
                enhanced_image = self._adjust_gamma(enhanced_image, 1.2)
                self._enh_image = enhanced_image
            else:
                print("Image is not in RGB format. Cannot apply enhancement.")

        return self._enh_image

    @staticmethod
    def _clahe_equalized(image):
        # Convert PIL Image to numpy array
        img = np.array(image)

        # Convert BGR to RGB if necessary
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Apply CLAHE to each channel separately
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_equalized = np.empty(img.shape, dtype=np.uint8)
        for c in range(3):
            img_equalized[:, :, c] = clahe.apply(img[:, :, c])

        # Convert back to PIL Image
        enhanced_image = Image.fromarray(img_equalized)
        return enhanced_image

    @staticmethod
    def _adjust_gamma(image, gamma=1.0):
        # Convert PIL Image to numpy array
        img = np.array(image)

        # Apply gamma correction to each channel separately
        img_gamma_corrected = np.empty(img.shape, dtype=np.uint8)
        for c in range(3):
            img_gamma_corrected[:, :, c] = cv2.LUT(img[:, :, c], FundusImage._create_gamma_lut(gamma))

        # Convert back to PIL Image
        enhanced_image = Image.fromarray(img_gamma_corrected)
        return enhanced_image

    @staticmethod
    def _create_gamma_lut(gamma=1.0):
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8)
        return table

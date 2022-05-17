import cv2 as cv
import numpy as np

from config import conf

"""
reference： https://www.researchgate.net/profile/Gabriel-Sanchez-Perez/publication/262371199_Explicit_image_detection_using_YCbCr_space_color_model_as_skin_detection/links/549839cf0cf2519f5a1dd966/Explicit-image-detection-using-YCbCr-space-color-model-as-skin-detection.pdf
"""

skin_ycrcb_mint = np.array((0, 133, 77))
skin_ycrcb_maxt = np.array((255, 173, 127))


def detect_skin(image: np.ndarray) -> np.ndarray:
    """
    Detection skin region in YCrCb space
    :param image: RGB image
    :return: skin mask for this image, dtype=uint8
    """
    im_ycrcb = cv.cvtColor(image, cv.COLOR_RGB2YCrCb)

    skin_ycrcb = cv.inRange(im_ycrcb, skin_ycrcb_mint, skin_ycrcb_maxt)

    skin_mask = np.dstack((skin_ycrcb, skin_ycrcb, skin_ycrcb))
    skin_mask = cv.GaussianBlur(skin_mask, (21, 21), 0)
    if conf.getboolean('Common', 'Debug'):
        cv.imshow('skin mask', skin_mask)
    return skin_mask

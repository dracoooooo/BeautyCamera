import cv2 as cv
import numpy as np

from config import conf

"""
reference: https://ieeexplore.ieee.org/document/767122
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

import cv2 as cv
import numpy as np

from config import conf


class Region:
    def __init__(self, image: np.ndarray, mask: np.ndarray):
        """
        :param image: RGB image
        :param mask: image mask, dtype=uint8
        """
        self.image_rgb = image
        self.image_hsv = cv.cvtColor(image, cv.COLOR_RGB2HSV)
        self.image_rgb_tmp = np.array(self.image_rgb)
        self.image_hsv_tmp = np.array(self.image_hsv)
        self.image_rgb_float = np.array(self.image_rgb, dtype=np.float64)
        self.image_hsv_float = np.array(self.image_hsv, dtype=np.float64)
        self.mask = mask.astype(np.float64)
        self.mask = self.mask / 255
        self.patch = 1 - self.mask

    def generate_result(self, new_image: np.ndarray, rate: float):
        """
        :param new_image: processed RGB image
        :param rate: 0-1
        :return: none
        """
        # blur_image = cv.GaussianBlur(new_image, (3, 3), 0)
        new_image_masked_float = new_image * self.mask
        raw_image_patch_float = self.image_rgb * self.patch
        raw_image_masked_float = self.image_rgb * self.mask

        image_rgb = np.minimum(
            rate * new_image_masked_float + (1 - rate) * raw_image_masked_float + raw_image_patch_float, 255).astype(
            np.uint8)

        np.copyto(self.image_rgb, image_rgb)
        self.update()

    def update(self):
        """
        Update images, assume that image_rgb is new
        """
        cv.cvtColor(self.image_rgb, cv.COLOR_RGB2HSV, self.image_hsv)
        self.image_rgb_tmp = np.array(self.image_rgb)
        self.image_hsv_tmp = np.array(self.image_hsv)

    def brighten(self, rate: int = 10):
        """
        V += rate
        :param rate: int, 0-40
        :return: None
        """
        self.image_hsv_tmp[:, :, 1] = cv.add(self.image_hsv_tmp[:, :, 1], rate)
        new_image = cv.cvtColor(self.image_hsv_tmp, cv.COLOR_HSV2RGB)
        if conf.getboolean('Common', 'Debug'):
            cv.imshow('brighten', cv.cvtColor(new_image, cv.COLOR_RGB2BGR))
        self.generate_result(new_image, 1)

    def lighten(self, rate: int = 10):
        """
        V += rate
        :param rate: int, 0-40
        :return: None
        """
        self.image_hsv_tmp[:, :, 2] = cv.add(self.image_hsv_tmp[:, :, 2], rate)
        new_image = cv.cvtColor(self.image_hsv_tmp, cv.COLOR_HSV2RGB)
        if conf.getboolean('Common', 'Debug'):
            cv.imshow('lighten', cv.cvtColor(new_image, cv.COLOR_RGB2BGR))
        self.generate_result(new_image, 1)

    def smooth(self, rate: int = 5):
        """
        Use bilateralFilter to smooth image
        :param rate: 0-10
        :return: None
        """
        new_image = cv.bilateralFilter(self.image_rgb, d=rate, sigmaColor=rate * 4, sigmaSpace=rate)
        if conf.getboolean('Common', 'Debug'):
            cv.imshow('smooth', cv.cvtColor(new_image, cv.COLOR_RGB2BGR))
        self.generate_result(new_image, 1.0)

    def sharpen(self, rate: float):
        """
        USM
        :param rate: float, 0-1
        :return: None
        """
        blur_img = cv.GaussianBlur(self.image_rgb, (0, 0), 5)
        new_image = cv.addWeighted(self.image_rgb, 1 + rate, blur_img, - rate, 0)
        if conf.getboolean('Common', 'Debug'):
            cv.imshow('sharpen', cv.cvtColor(new_image, cv.COLOR_RGB2BGR))
        self.generate_result(new_image, 1.0)

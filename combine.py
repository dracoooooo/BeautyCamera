import cv2 as cv
import numpy as np


def combine_two_color_images_with_anchor(image1: np.ndarray, image2: np.ndarray, mask: np.ndarray, anchor_y: int,
                                         anchor_x: int, alpha: float = 1) -> np.ndarray:
    """
    Combine two images.
    :param image1: foreground
    :param image2: background
    :param mask: foreground mask, same size as image1
    :param anchor_y: Y coordinate relative to image2
    :param anchor_x: X coordinate relative to image2
    :param alpha: transparency, default 1
    :return: combined image, same size as image2
    """
    foreground, background = image1.copy(), image2.copy()
    background_height = background.shape[0]
    background_width = background.shape[1]
    foreground_height = foreground.shape[0]
    foreground_width = foreground.shape[1]

    # Compute start and end position on 2 images
    background_start_y = anchor_y - round(foreground_height / 2) if anchor_y - round(foreground_height / 2) >= 0 else 0
    background_start_x = anchor_x - round(foreground_width / 2) if anchor_x - round(foreground_width / 2) >= 0 else 0
    background_end_y = anchor_y + int(foreground_height / 2) if anchor_y + int(
        foreground_height / 2) <= background_height else background_height
    background_end_x = anchor_x + int(foreground_width / 2) if anchor_x + int(
        foreground_width / 2) <= background_width else background_width

    foreground_start_y = round(foreground_height / 2) - anchor_y if round(foreground_height / 2) - anchor_y >= 0 else 0
    foreground_start_x = round(foreground_width / 2) - anchor_x if round(foreground_width / 2) - anchor_x >= 0 else 0
    foreground_end_y = int(foreground_height / 2) - anchor_y + background_height if int(
        foreground_height / 2) - anchor_y + background_height <= foreground_height else foreground_height
    foreground_end_x = int(foreground_width / 2) - anchor_x + background_width if int(
        foreground_width / 2) - anchor_x + background_width <= foreground_width else foreground_width

    assert background_end_x - background_start_x == foreground_end_x - foreground_start_x
    assert background_end_y - background_start_y == foreground_end_y - foreground_start_y

    # Do composite at specified location
    blended_portion = cv.addWeighted(
        foreground[foreground_start_y: foreground_end_y, foreground_start_x: foreground_end_x, :],
        alpha,
        background[background_start_y: background_end_y, background_start_x: background_end_x, :],
        1 - alpha,
        0,
        )
    blended_portion = np.where(mask[foreground_start_y: foreground_end_y, foreground_start_x: foreground_end_x, :] == 1,
                               blended_portion,
                               background[background_start_y: background_end_y, background_start_x: background_end_x,
                               :])
    background[background_start_y: background_end_y, background_start_x: background_end_x, :] = blended_portion
    return background

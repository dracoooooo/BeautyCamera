import cv2 as cv
import numpy as np

from config import conf

"""
reference: https://learnopencv.com/head-pose-estimation-using-opencv-and-dlib/
"""

# General 3D model points.
model_points = np.array([
    (0.0, 0.0, 0.0),  # Nose tip
    (0.0, -330.0, -65.0),  # Chin
    (-225.0, 170.0, -135.0),  # Left eye left corner
    (225.0, 170.0, -135.0),  # Right eye right corner
    (-150.0, -150.0, -125.0),  # Left Mouth corner
    (150.0, -150.0, -125.0)  # Right mouth corner
])

pose_points = [29, 8, 36, 45, 48, 54]


def get_head_pose(landmarks: np.ndarray, image: np.ndarray) -> np.ndarray:
    """
    :param landmarks: dlib facial landmarks, at least 68
    :param image: raw image
    :return: rvec
    """
    # Pick 6 points from landmarks array
    image_points = landmarks[pose_points].astype(np.float64)

    # Camera internals
    size = image.shape
    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype=np.float64
    )

    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv.solvePnP(model_points, image_points, camera_matrix, dist_coeffs,
                                                                 flags=cv.SOLVEPNP_UPNP)

    if conf.getboolean('Common', 'Debug'):
        # Show head pose
        (nose_end_point2D, jacobian) = cv.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector,
                                                        translation_vector, camera_matrix, dist_coeffs)
        p1 = (int(image_points[0][0]), int(image_points[0][1]))
        p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
        cv.line(image, p1, p2, (255, 0, 0), 2)
        cv.imshow("head pose", cv.cvtColor(image, cv.COLOR_RGB2BGR))

    return rotation_vector

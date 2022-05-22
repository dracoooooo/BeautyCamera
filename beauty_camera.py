import cv2 as cv
import dlib
import numpy as np
from imutils import face_utils

from combine import combine_two_color_images_with_anchor
from config import conf
from effect import SpecialEffect
from headPose import get_head_pose
from region import Region
from skin import detect_skin

organs_name = ['jaw', 'mouth', 'nose', 'left eye', 'right eye', 'left brow', 'right brow', 'forehead']

organs_points = [list(range(0, 17)), list(range(48, 61)), list(range(27, 36)), list(range(42, 48)), list(range(36, 42)),
                 list(range(22, 27)), list(range(17, 22)), list(range(68, 81)).append([0, 16])]


class BeautyCamera:
    def __init__(self, frame: np.ndarray = None):
        """
        :param frame: BGR image
        :param effect: a special effect applied to the image
        """
        # Read config
        effects = {}
        for section in conf.sections():
            if section != 'Common' and section != 'Beautify':
                effect = SpecialEffect(conf.get(section, 'Path'), conf.getint(section, 'Landmark'))
                effects[section] = effect
        self.effects = effects

        effect_name = conf.get('Common', 'Effect')
        if effect_name != 'None':
            self.effect = effects[effect_name]
        else:
            self.effect = None

        self.frame = frame
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(conf.get('Common', 'PredictorPath'))

        if frame is not None:
            mask = np.ones_like(frame, dtype='uint8') * 255
            self.whole_image = Region(cv.cvtColor(frame, cv.COLOR_BGR2RGB), mask)

    def set_frame(self, frame: np.ndarray):
        self.frame = frame
        mask = np.ones_like(frame, dtype='uint8') * 255
        self.whole_image = Region(cv.cvtColor(frame, cv.COLOR_BGR2RGB), mask)

    def process(self):
        if self.frame is None:
            pass

        show_image = eval(conf.get('Common', 'ShowImage'))
        debug = conf.getboolean('Common', 'Debug')

        if show_image:
            cv.imshow('raw', self.frame)

        frameBGR = self.frame
        frameRGB = cv.cvtColor(frameBGR, cv.COLOR_BGR2RGB)

        region_mask = detect_skin(frameRGB)

        if debug:
            cv.imshow('skin mask', region_mask)

        skin = Region(frameRGB, region_mask)

        skin.lighten(conf.getint('Beautify', 'SkinLighten'))
        skin.sharpen(conf.getfloat('Beautify', 'SkinSharpen'))
        skin.smooth(conf.getint('Beautify', 'SkinSmooth'))

        dets = self.detector(frameRGB, 1)

        faces = []

        for k, d in enumerate(dets):
            (x, y, w, h) = face_utils.rect_to_bb(d)
            shape = self.predictor(frameRGB, d)
            shape_np = face_utils.shape_to_np(shape)

            if debug:
                point_color = eval(conf.get('Common', 'PointColor'))
                cv.rectangle(frameRGB, (x, y), (x + w, y + h), point_color)
                for (x, y) in shape_np:
                    cv.circle(frameRGB, (x, y), 2, point_color)

            face = {}
            face['landmarks'] = shape_np

            for t in range(len(organs_points)):
                hull = cv.convexHull(shape_np[organs_points[t]])
                # for i in range(len(hull)):
                #     cv.line(frameRGB, tuple(hull[i][0]), tuple(hull[(i + 1) % len(hull)][0]), RGB_GREEN)
                region_mask = np.zeros_like(frameRGB, dtype=np.uint8)
                cv.fillConvexPoly(region_mask, hull, (255, 255, 255))
                region_mask = cv.GaussianBlur(region_mask, (25, 25), 0)
                if debug:
                    cv.imshow(organs_name[t], region_mask)
                face[organs_name[t]] = Region(frameRGB, region_mask)

            faces.append(face)

        for face in faces:
            face['left eye'].lighten(conf.getint('Beautify', 'EyeLighten'))
            face['right eye'].lighten(conf.getint('Beautify', 'EyeLighten'))
            face['mouth'].brighten(conf.getint('Beautify', 'MouthBrighten'))

            if self.effect is not None:
                landmarks = face['landmarks']
                r = get_head_pose(landmarks, frameRGB)
                distance = np.linalg.norm(landmarks[0] - landmarks[16])
                self.effect.rotate(r)
                self.effect.scale(distance / self.effect.size)
                effect_frame = self.effect.get_frame()
                frameRGB = combine_two_color_images_with_anchor(effect_frame, frameRGB, self.effect.get_mask(),
                                                                landmarks[self.effect.landmark][1],
                                                                landmarks[self.effect.landmark][0])

        self.whole_image.brighten(conf.getint('Beautify', 'ImageLighten'))

        frameBGR = cv.cvtColor(frameRGB, cv.COLOR_RGB2BGR)

        if show_image:
            cv.imshow('frame', frameBGR)

        np.copyto(self.frame, frameBGR)

import os

import cv2 as cv
from tqdm import trange

from beauty_camera import BeautyCamera
from config import conf

if __name__ == '__main__':
    bc = BeautyCamera()

    mode = conf.get('Common', 'Mode')

    if mode == 'Camera':
        # Open camera
        video = cv.VideoCapture(0)

        if not video.isOpened():
            print("Cannot open camera")
            exit()

        while True:
            ret, frameBGR = video.read()
            if not ret:
                print("Cannot receive frame")
                exit()

            bc.set_frame(frameBGR)
            bc.process()

            if cv.waitKey(1) == ord('q'):
                break

    elif mode == 'Image':
        image_path = conf.get('Common', 'InputPath')
        frame = cv.imread(image_path)
        bc.set_frame(frame)
        bc.process()

        filename = os.path.basename(image_path)
        output_path = conf.get('Common', 'OutputPath')
        cv.imwrite(output_path + filename, frame)

        cv.waitKey(0)

    elif mode == 'Video':
        video_path = conf.get('Common', 'InputPath')
        video = cv.VideoCapture(video_path)

        if not video.isOpened():
            print("Error reading video file")
            exit()

        frame_width = int(video.get(3))
        frame_height = int(video.get(4))

        size = (frame_width, frame_height)
        fps = video.get(cv.CAP_PROP_FPS)
        length = int(video.get(cv.CAP_PROP_FRAME_COUNT))

        filename = os.path.basename(video_path)
        output_path = conf.get('Common', 'OutputPath')

        writer = cv.VideoWriter(output_path + filename + '.avi',
                                cv.VideoWriter_fourcc(*'MJPG'),
                                fps, size)

        for i in trange(length):
            ret, frame = video.read()

            if not ret:
                print("Cannot receive frame")
                break

            bc.set_frame(frame)
            bc.process()

            writer.write(frame)

            if cv.waitKey(1) == ord('q'):
                break

    else:
        print('Mode error')
        exit()

import cv2
import numpy as np


class History(object):
    def __init__(self, img_size, history_length, format='NCHW'):
        self._img_size = img_size
        self._history_length = history_length
        self._format = format
        self._state = np.zeros((history_length, img_size[0], img_size[1]))

    def add(self, img):
        proc_img = self._image_preprocessing(img)
        self._state = np.roll(self._state, 1)
        self._state[0] = proc_img

    def reset(self):
        self._state = np.zeros((
            self._history_length, self._img_size[0], self._img_size[1]))

    def _image_preprocessing(self, img):
        grayscaled = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(grayscaled, self._img_size)

        return resized

    @property
    def state(self):
        if self._format == "NHWC":
            return self._state.transpose(1, 2, 0)
        return self._state

#! /usr/bin/env python

import cv2
import cvlib as cv
from math import *

class Detection:
    def detect(self, cv_image):
        detects, labels, confs = cv.detect_common_objects(cv_image, model='yolov3-tiny', enable_gpu=True)
        indices = cv2.dnn.NMSBoxes(detects, confs, score_threshold=0.2, nms_threshold=0.5)

        bboxes = []
        for i in indices:
            i = i[0]
            bbox = detects[i]
            bboxes.append(bbox)
        return bboxes
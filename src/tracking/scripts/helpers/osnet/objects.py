#! /usr/bin/env python

import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import time
from scipy.spatial import distance as dist
import rospy
import rospkg
rospack = rospkg.RosPack()

model = os.path.join(rospack.get_path("tracking"), "scripts/helpers/mars/models/", "mars-small128.pb")

class OSFeatures:
    def __init__(self):
        self.model = model

    def __preProcess(self, bboxes):
        #Convert tlrb to tlwh
        boxes = np.array(bboxes)
        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
        return boxes
    
    def extractBBoxFeatures(self, img, bboxes, target_id=0):
        print(bboxes)
        bbox = self.__preProcess([bboxes[target_id]])
        
    
    def extractBBoxesFeatures(self, img, bboxes):
        bboxes = self.__preProcess(bboxes)
        
#! /usr/bin/env python

import numpy as np
import os
import time
from scipy.spatial import distance as dist
import rospy
import rospkg
rospack = rospkg.RosPack()

model = os.path.join(rospack.get_path("tracking"), "scripts/helpers/deep_features/model/", "mars-small128.pb")

class DeepFeatures:
    def __init__(self):
        self.model = model
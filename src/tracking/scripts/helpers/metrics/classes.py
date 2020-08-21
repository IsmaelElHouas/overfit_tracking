#!/usr/bin/env python3
import numpy
import pandas as pd
import sys
import os
import rospy
import rospkg
rospack = rospkg.RosPack()

save_path = os.join(rospack.get_path('tracking'), '/scripts/helpers/metrics/test_learn.csv')

class Metrics():
    def __init__(self):
        self.data = []

    def collect_data(self, frame_count, id, distances_o, distances_m, histogram):
        self.data.append({'FRAME': frame_count, 'ID': id, 'OSNET': distances_o , 'MOBILENET': distances_m, 'HISTOGRAM': histogram})

    def save_data(self):
        df = pd.DataFrame(self.data)
        df.to_csv(save_path, encoding='utf-8')

    
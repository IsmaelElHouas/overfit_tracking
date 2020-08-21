#!/usr/bin/env python3

import cv2
from cv_bridge import CvBridge, CvBridgeError
from copy import deepcopy
import numpy as np

from math import *
import os
import rospy
import rospkg
rospack = rospkg.RosPack()

from std_msgs.msg import Int8, String
from sensor_msgs.msg import Image
from tracking.msg import BBox, BBoxes

# helpers
from helpers.detection import Detection
detection = Detection()

from helpers.deep_feature_tracking import *
dft = DeepFeatures()

from helpers.metrics import Metrics
metrics = Metrics()

roi_dist = 400 # To-do: dynamic
feature_dist = 0.4
neighbor_dist = 0.15

height = 720
width = 960 

class Detect:
    def __init__(self):
        rospy.init_node('detect_node', anonymous=True)
        rate = rospy.Rate(30)
        
        self.bridge = CvBridge()
        self.frame = None
        self.tracking_bbox_features = None
        self.prev_target_cent = None
        self.prev_target_features = None

        rospy.Subscriber('/stream/image', Image, self.img_callback)
        
        frame_count = 0
        target_id = 0
        distances_o = None
        distances_m = None
        distances_h = None

        while not rospy.is_shutdown():
            if self.frame is not None:      
                frame = deepcopy(self.frame)
                centroids, bboxes = detection.detect(frame)

                if len(centroids) == 0:
                    break

                if frame_count == 0:
                    dft.extractBBoxFeatures(frame, bboxes, target_id) 
                    self.prev_target_cent = centroids[target_id]
                    print("catch once")
                else:
                    if self.prev_target_cent is not None:
                        #centroids_roi, bboxes_roi = self.__roi(centroids, bboxes)

                        if len(centroids) > 0:
                            # extract features of bboxes
                            tracking_id, distances_o, distances_m, distances_h = dft.matchBoundingBoxes(self.frame, bboxes)
                            
                            metrics.collect_data(frame_count, tracking_id, distances_o, distances_m, distances_h)

                            if tracking_id != -1:
                                target_cent = centroids[tracking_id]
                                self.prev_target_cent = target_cent
                                cv2.rectangle(frame, (target_cent[0]-20, target_cent[1]-40), (target_cent[0]+20, target_cent[1]+40), (255,0,0), 1)
                                cv2.putText(frame, str(frame_count), (target_cent[0]-20, target_cent[1]-40), cv2.FONT_HERSHEY_PLAIN, 10, (0,0,255), 3)

                frame_count = frame_count + 1
                cv2.imshow("", frame)
                cv2.waitKey(1)
            metrics.save_data()    
            rate.sleep()
            

    def img_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data)
        except CvBridgeError as e:
            print(e)
        self.frame = cv_image


if __name__ == '__main__':
    try:
        Detect()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()
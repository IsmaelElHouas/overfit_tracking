#!/usr/bin/env python

import cv2
from cv_bridge import CvBridge, CvBridgeError
from copy import deepcopy
import numpy as np
from scipy.spatial import distance as dist

import os
import rospy
import rospkg
rospack = rospkg.RosPack()

from std_msgs.msg import Int8, String
from sensor_msgs.msg import Image
from tracking.msg import BBox, BBoxes

# Helpers
from helpers.cvlib import Detection
detection = Detection()

from helpers.deep_features import DeepFeatures
deep_features = DeepFeatures()


class Detect:
    def __init__(self):
        rospy.init_node('detect_node', anonymous=True)
        rate = rospy.Rate(30)
        
        self.bridge = CvBridge()
        self.frame = None
        self.tracking_bbox_features = []       

        rospy.Subscriber('/stream/image', Image, self.img_callback)
        bboxes_pub = rospy.Publisher('/detection/bboxes', BBoxes, queue_size=10)
        
        frame_count = 0
        track_id = 0
        while not rospy.is_shutdown():
            if self.frame is not None:      
                frame = deepcopy(self.frame)
                centroids, bboxes = detection.detect(frame)

                if frame_count == 0:
                    self.tracking_bbox_features = deep_features.extractBBoxFeatures(frame, bboxes, track_id)
                else:
                    bboxes_features = deep_features.extractBBoxesFeatures(frame, bboxes)
                    features_distance = dist.cdist(self.tracking_bbox_features, bboxes_features, "cosine")
                    print(features_distance)


                if len(centroids) != 0:
                    for cent in centroids:
                        cv2.rectangle(frame, (cent[0]-20, cent[1]-40), (cent[0]+20, cent[1]+40), (255,0,0), 1)

                cv2.imshow("", frame)
                cv2.waitKey(1)
                frame_count = frame_count + 1

            rate.sleep()

    def img_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data)
        except CvBridgeError as e:
            print(e)
        self.frame = cv_image

    #Tracking functions
    def __extractTrackingBBoxFeatures(self, bboxes, tracking_id):
        bbox_features = deep_features.extractBBoxFeatures(self.frame, bboxes, tracking_id=tracking_id)
        return bbox_features
    
    def __calcFeaturesDistance(self, bboxes):
        bboxes_features = deep_features.extractBBoxesFeatures(self.frame, bboxes)
        features_distance = dist.cdist(self.tracking_bbox_features, bboxes_features, "cosine")
        return features_distance


if __name__ == '__main__':
    try:
        Detect()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()
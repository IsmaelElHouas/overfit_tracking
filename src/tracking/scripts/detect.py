#!/usr/bin/env python

import cv2
from copy import deepcopy
import numpy as np
import os
import rospy
import rospkg
rospack = rospkg.RosPack()
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from tracking.msg import BBox, BBoxes

# Helpers
from helpers.cvlib import Detection
detection = Detection()


class Detect:
    def __init__(self):
        rospy.init_node('detect_node', anonymous=True)
        rate = rospy.Rate(10)
        
        self.bridge = CvBridge()
        self.frame = None

        rospy.Subscriber('/stream/image', Image, self.img_callback)
        bboxes_pub = rospy.Publisher('/detection/bboxes', BBoxes, queue_size=10)
        
        while not rospy.is_shutdown():
            if self.frame is not None:
                frame = deepcopy(self.frame)
                bboxes = detection.detect(frame)
                msgs = self.__bboxesMsgProcess(bboxes)
                bboxes_pub.publish(msgs)
            rate.sleep()

    def img_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data)
        except CvBridgeError as e:
            print(e)
        self.frame = cv_image

    def __bboxesMsgProcess(self, bboxes):
        msgs = BBoxes()
        for i in range(len(bboxes)):
            msg = BBox()
            msg.bbox = bboxes[i]
            msgs.bboxes.append(msg)
        return msgs


if __name__ == '__main__':
    try:
        Detect()
    except rospy.ROSInterruptException:
        pass
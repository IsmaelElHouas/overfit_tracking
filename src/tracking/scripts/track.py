#! /usr/bin/env python

import cv2
import numpy as np
import rospy
import os
import rospkg
rospack = rospkg.RosPack()

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

# Helpers
from helpers.cvlib import Detection
detection = Detection()

from helpers.keyboard import Keyboard
keyboard = Keyboard()

def Track():
    def __init__(self):
        rospy.init_node('track_node', anonymous=True)
        rate = rospy.Rate(30)

        self.bridge = CvBridge()
        self.frame = None

        rospy.Subscriber('/stream/image', Image, self.img_callback)

        while not rospy.is_shutdown():
            key = keyboard.listener()
            print(key)
            rate.sleep()

    def img_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data)
        except CvBridgeError as e:
            print(e)
        self.frame = cv_image


if __name__ == '__main__':
    try:
        Track()
    except rospy.ROSInterruptException:
        pass
        
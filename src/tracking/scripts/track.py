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

def Stream():
    rospy.init_node('stream_node', anonymous=True)
    rate = rospy.Rate(30)

    video_path = os.path.join(rospack.get_path("tracking"), "scripts/input", "cross_id.avi")
    cap = cv2.VideoCapture(video_path)
    
    frame_counter = 0
    targets = [[344, 516, 412, 780], [914, 526, 972, 644], [671, 555, 735, 715]]
    while cap.isOpened() and not rospy.is_shutdown():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_counter < 10:
            centroids, bboxes = detection.detect(frame)
            print(bboxes)
        else:
            break

        frame_counter += 1
        rate.sleep()


    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        Stream()
    except rospy.ROSInterruptException:
        pass
        
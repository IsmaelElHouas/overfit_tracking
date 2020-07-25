#! /usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float64

from copy import deepcopy
from cv_bridge import CvBridge, CvBridgeError
import cv2

from math import *
import numpy as np
import time

# helper
from helpers.openpose import OpenPoseVGG
openpose = OpenPoseVGG()

POSE_PAIRS = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7],
              [1,8], [8,9], [9,10], [1,11], [11,12], [12,13],
              [1,0], [0,14], [14,16], [0,15], [15,17],
              [2,17], [5,16] ]
colors = [ [0,100,255], [0,100,255], [0,255,255], [0,100,255], [0,255,255], [0,100,255],
         [0,255,0], [255,200,100], [255,0,255], [0,255,0], [255,200,100], [255,0,255],
         [0,0,255], [255,0,0], [200,200,0], [255,0,0], [200,200,0], [0,0,0]]

class Detect:
    def __init__(self):
        rospy.init_node('detect_node', anonymous=True)
        self.rate = rospy.Rate(10)

        self.bridge_object = CvBridge()
        self.frame = None

        rospy.Subscriber('/stream/image', Image, self.img_callback)

        while not rospy.is_shutdown():
            if self.frame is not None:
                frame = deepcopy(self.frame)

                features = openpose.detectFeatures(frame)
                # print(features.shape)
                
                personwiseKeypoints,  keypoints_list= openpose.detectPersonwiseKeypoints(frame)
                for i in range(18):
                    for n in range(len(personwiseKeypoints)):
                        index = personwiseKeypoints[n][np.array(POSE_PAIRS[i])]
                        if -1 in index:
                            continue
                        B = np.int32(keypoints_list[index.astype(int), 0])
                        A = np.int32(keypoints_list[index.astype(int), 1])
                        if i==0:
                            cv2.putText(frame, str(n), (B[0], A[0]-50), cv2.FONT_HERSHEY_PLAIN, 1.0, colors[n], 2)
                        cv2.line(frame, (B[0], A[0]), (B[1], A[1]), colors[i], 3, cv2.LINE_AA)
                
                # detected_keypoints = openpose.detectKeypoints(frame)
                
                # for i in range(18):
                #     for j in range(len(detected_keypoints[i])):
                #         cv2.circle(frame, detected_keypoints[i][j][0:2], 3, [0,0,255], -1, cv2.LINE_AA)

                cv2.imshow("", frame)
                cv2.waitKey(1)
            self.rate.sleep()
    
    def img_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data)
        except CvBridgeError as e:
            print(e)
        self.frame = cv_image


def main():
    try:
        Detect()
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
    

if __name__ == '__main__':
    main()
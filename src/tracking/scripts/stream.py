#! /usr/bin/env python

import cv2
import numpy as np
import rospy
import os
import rospkg
rospack = rospkg.RosPack()

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

def Stream():
    rospy.init_node('stream_node', anonymous=True)
    rate = rospy.Rate(30)

    img_pub = rospy.Publisher('stream/image', Image, queue_size=10)

    video_path = os.path.join(rospack.get_path("tracking"), "scripts/input", "cross_id.avi")
    cap = cv2.VideoCapture(video_path)
    
    frame_counter = 0
    while cap.isOpened() and not rospy.is_shutdown():
        ret, frame = cap.read()
        frame_counter += 1
        if not ret:
            break
        
        # #Looping the video
        # if frame_counter == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        #     frame_counter = 0
        #     cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
        frame = np.uint8(frame)
        frame_msg = CvBridge().cv2_to_imgmsg(frame, encoding="passthrough")
        img_pub.publish(frame_msg)

        rate.sleep()

    cap.release()


if __name__ == '__main__':
    try:
        Stream()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()

#! /usr/bin/env python

import cv2
import numpy as np
import os
import rospy
import rospkg
import time
from math import *

rospack = rospkg.RosPack()
openpose_folder = os.path.join(rospack.get_path("tracking"), "scripts/helpers/openpose/models/")
net = cv2.dnn.readNetFromTensorflow(openpose_folder + "graph_opt.pb")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
nPoints = 18
threshold = 0.1
inputSize = 300

keypointsMapping = ['Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho', 
                    'L-Elb', 'L-Wr', 'R-Hip', 'R-Knee', 'R-Ank', 'L-Hip', 
                    'L-Knee', 'L-Ank', 'R-Eye', 'L-Eye', 'R-Ear', 'L-Ear']

POSE_PAIRS = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7],
              [1,8], [8,9], [9,10], [1,11], [11,12], [12,13],
              [1,0], [0,14], [14,16], [0,15], [15,17],
              [2,17], [5,16]]

mapIdx = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44], 
          [19,20], [21,22], [23,24], [25,26], [27,28], [29,30], 
          [47,48], [49,50], [53,54], [51,52], [55,56], 
          [37,38], [45,46]]

colors = [ [0,100,255], [0,100,255], [0,255,255], [0,100,255], [0,255,255], [0,100,255],
         [0,255,0], [255,200,100], [255,0,255], [0,255,0], [255,200,100], [255,0,255],
         [0,0,255], [255,0,0], [200,200,0], [255,0,0], [200,200,0], [0,0,0]]

class OpenPose():
    def __init__(self):
        self.frameWidth = 0
        self.frameHeight = 0
        self.detected_keypoints = []
        

    def detect(self, img):
        self.frameWidth = img.shape[1]
        self.frameHeight = img.shape[0]
        keypoints_list = np.zeros((0,3))
        keypoint_id = 0

        net.setInput(cv2.dnn.blobFromImage(img, 1.0, (inputSize, inputSize), (127.5, 127.5, 127.5), swapRB=True, crop=False))
        output = net.forward()
        output_points = output[:, :nPoints, :, :]

        H = output_points.shape[2]
        W = output_points.shape[3]

        points = []
        for i in range(nPoints):
            probMap = output_points[0, i, :, :]
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
            x = (self.frameWidth * point[0]) / W
            y = (self.frameHeight * point[1]) / H

            if (prob > threshold):
                points.append((int(x), int(y)))
            else :
                points.append(None)

            keypoints = self.getKeypoints(probMap, threshold)
            print("Keypoints - {} : {}".format(keypointsMapping[i], keypoints))
            keypoints_with_id = []
            for i in range(len(keypoints)):
                keypoints_with_id.append(keypoints[i] + (keypoint_id,))
                keypoints_list = np.vstack([keypoints_list, keypoints[i]])
                keypoint_id += 1

            self.detected_keypoints.append(keypoints_with_id)

        
        valid_pairs, invalid_pairs = self.getValidPairs(output)
        print(valid_pairs)
        return points

    # Find the Keypoints using Non Maximum Suppression on the Confidence Map
    def getKeypoints(self, probMap, threshold=0.1):
        mapSmooth = cv2.GaussianBlur(probMap,(3,3),0,0)
        mapMask = np.uint8(mapSmooth>threshold)
        keypoints = []
        
        #find the blobs
        contours, _ = cv2.findContours(mapMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        #for each blob find the maxima
        for cnt in contours:
            blobMask = np.zeros(mapMask.shape)
            blobMask = cv2.fillConvexPoly(blobMask, cnt, 1)
            maskedProbMap = mapSmooth * blobMask
            _, maxVal, _, maxLoc = cv2.minMaxLoc(maskedProbMap)
            keypoints.append(maxLoc + (probMap[maxLoc[1], maxLoc[0]],))

        return keypoints

    # Find valid connections between the different joints of a all persons present
    # Find valid connections between the different joints of a all persons present
    def getValidPairs(self, output):
        valid_pairs = []
        invalid_pairs = []
        n_interp_samples = 10
        paf_score_th = 0.1
        conf_th = 0.7
        # loop for every POSE_PAIR
        for k in range(len(mapIdx)):
            # A->B constitute a limb
            pafA = output[0, mapIdx[k][0], :, :]
            pafB = output[0, mapIdx[k][1], :, :]
            pafA = cv2.resize(pafA, (self.frameWidth, self.frameHeight))
            pafB = cv2.resize(pafB, (self.frameWidth, self.frameHeight))

            # Find the keypoints for the first and second limb
            candA = self.detected_keypoints[POSE_PAIRS[k][0]]
            candB = self.detected_keypoints[POSE_PAIRS[k][1]]
            nA = len(candA)
            nB = len(candB)

            # If keypoints for the joint-pair is detected
            # check every joint in candA with every joint in candB 
            # Calculate the distance vector between the two joints
            # Find the PAF values at a set of interpolated points between the joints
            # Use the above formula to compute a score to mark the connection valid
            
            if( nA != 0 and nB != 0):
                valid_pair = np.zeros((0,3))
                for i in range(nA):
                    max_j=-1
                    maxScore = -1
                    found = 0
                    for j in range(nB):
                        # Find d_ij
                        d_ij = np.subtract(candB[j][:2], candA[i][:2])
                        norm = np.linalg.norm(d_ij)
                        if norm:
                            d_ij = d_ij / norm
                        else:
                            continue
                        # Find p(u)
                        interp_coord = list(zip(np.linspace(candA[i][0], candB[j][0], num=n_interp_samples),
                                                np.linspace(candA[i][1], candB[j][1], num=n_interp_samples)))
                        # Find L(p(u))
                        paf_interp = []
                        for k in range(len(interp_coord)):
                            paf_interp.append([pafA[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))],
                                            pafB[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))] ]) 
                        # Find E
                        paf_scores = np.dot(paf_interp, d_ij)
                        avg_paf_score = sum(paf_scores)/len(paf_scores)
                        
                        # Check if the connection is valid
                        # If the fraction of interpolated vectors aligned with PAF is higher then threshold -> Valid Pair  
                        if ( len(np.where(paf_scores > paf_score_th)[0]) / n_interp_samples ) > conf_th :
                            if avg_paf_score > maxScore:
                                max_j = j
                                maxScore = avg_paf_score
                                found = 1
                    # Append the connection to the list
                    if found:            
                        valid_pair = np.append(valid_pair, [[candA[i][3], candB[max_j][3], maxScore]], axis=0)

                # Append the detected connections to the global list
                valid_pairs.append(valid_pair)
            else: # If no keypoints are detected
                print("No Connection : k = {}".format(k))
                invalid_pairs.append(k)
                valid_pairs.append([])
        print(valid_pairs)
        return valid_pairs, invalid_pairs
   
    # This function creates a list of keypoints belonging to each person
    # For each detected valid pair, it assigns the joint(s) to a person
    # It finds the person and index at which the joint should be added. This can be done since we have an id for each joint
    def getPersonwiseKeypoints(self, valid_pairs, invalid_pairs):
        # the last number in each row is the overall score 
        personwiseKeypoints = -1 * np.ones((0, 19))

        for k in range(len(mapIdx)):
            if k not in invalid_pairs:
                partAs = valid_pairs[k][:,0]
                partBs = valid_pairs[k][:,1]
                indexA, indexB = np.array(POSE_PAIRS[k])

                for i in range(len(valid_pairs[k])): 
                    found = 0
                    person_idx = -1
                    for j in range(len(personwiseKeypoints)):
                        if personwiseKeypoints[j][indexA] == partAs[i]:
                            person_idx = j
                            found = 1
                            break

                    if found:
                        personwiseKeypoints[person_idx][indexB] = partBs[i]
                        personwiseKeypoints[person_idx][-1] += keypoints_list[partBs[i].astype(int), 2] + valid_pairs[k][i][2]

                    # if find no partA in the subset, create a new subset
                    elif not found and k < 17:
                        row = -1 * np.ones(19)
                        row[indexA] = partAs[i]
                        row[indexB] = partBs[i]
                        # add the keypoint_scores for the two keypoints and the paf_score 
                        row[-1] = sum(keypoints_list[valid_pairs[k][i,:2].astype(int), 2]) + valid_pairs[k][i][2]
                        personwiseKeypoints = np.vstack([personwiseKeypoints, row])
        return personwiseKeypoints
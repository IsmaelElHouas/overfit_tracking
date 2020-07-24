#! /usr/bin/env python

import cv2
import numpy as np
import os
import rospy
import rospkg
rospack = rospkg.RosPack()

from PIL import Image as PIm
import torch
import torchvision
from torchreid.utils import FeatureExtractor

model = os.path.join(rospack.get_path("tracking"), "scripts/helpers/osnet/models/", "osnet_x1_0_imagenet.pth")

class OSFeatures:
    def __init__(self):
        self.extractor = FeatureExtractor(
            model_name='osnet_x1_0',
            model_path=model,
            device='cuda'
        )

        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((256, 128)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])
        ])

    def __preProcess(self, img, bbox):
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_array = np.asarray(img, dtype=np.uint8)
        pil_img = PIm.fromarray(img_array)
        cropped_img = pil_img.crop(bbox)
        cropped_img = self.transforms(cropped_img).cuda()
        cropped_img.unsqueeze_(0)
        return cropped_img

    def __featureExtractor(self, img_tensors):
        features = self.extractor(img_tensors)
        features = features.cpu().detach().numpy()
        return features
    
    def extractBBoxFeatures(self, img, bboxes, target_id=0):
        bbox = bboxes[target_id]
        cropped_img = self.__preProcess(img, bbox)
        features_bbox = self.__featureExtractor(cropped_img)
        return features_bbox
    
    def extractBBoxesFeatures(self, img, bboxes):
        cropped_imgs = []
        for bbox in bboxes:
            cropped_img = self.__preProcess(img, bbox)
            cropped_imgs.append(cropped_img[0])
        cropped_imgs = torch.stack(cropped_imgs)
        features_bboxes = self.__featureExtractor(cropped_imgs)
        return features_bboxes

        
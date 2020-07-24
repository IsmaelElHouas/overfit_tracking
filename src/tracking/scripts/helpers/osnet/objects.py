#! /usr/bin/env python

import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import time
from scipy.spatial import distance as dist
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

    def __preProcess(self, pil_frame, bbox):
        img = pil_frame.crop(bbox)
        img = self.transforms(img).cuda()
        img.unsqueeze_(0)
        return img
    
    def extractBBoxFeatures(self, img, bboxes, target_id=0):
        bbox = bboxes[target_id]
        frame = np.asarray(img, dtype=np.uint8)
        pil_frame = PIm.fromarray(frame)
        crop = self.__preProcess(pil_frame, bbox)
        features_torch_model = self.extractor(crop)
        features = features_torch_model.cpu().detach().numpy()
        return features
    
    def extractBBoxesFeatures(self, img, bboxes):
        frame = np.asarray(img, dtype=np.uint8)
        pil_frame = PIm.fromarray(frame)

        crops = []
        for bbox in bboxes:
            crop = self.__preProcess(pil_frame, bbox)
            crops.append(crop[0])
        crops = torch.stack(crops)

        features_torch_model = self.extractor(crops)
        features = features_torch_model.cpu().detach().numpy()

        return features

        
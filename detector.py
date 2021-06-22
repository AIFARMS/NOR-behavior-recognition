"""
Contains Detector class which loads the TSM model
"""

import sys
sys.path.append('models/TSM')

import cv2
import numpy as np
import imutils
import torch
import torchvision
import torch.nn.functional as F

from PIL import Image
from ops.models import TSN
from ops.transforms import *

from ptflops import get_model_complexity_info

X_CROP, Y_CROP = 350, 30


class TSM_detector():

    def __init__(self, modality, checkpoint, arena_mask_path):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TSN(2, 8, modality,
                    base_model='resnet18',
                    consensus_type='avg',
                    dropout=0.5,
                    img_feature_dim=256,
                    partial_bn=False,
                    pretrain='imagenet',
                    is_shift=True, shift_div=8, shift_place='blockres',
                    fc_lr5=False,
                    temporal_pool=False,
                    non_local=False)

        # Get Model complexity
        macs, params = get_model_complexity_info(model,
                    (24, 224, 224), as_strings=True, print_per_layer_stat=False, verbose=True)  # noqa: E128, E501
        print('---{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))

        # Define transforms
        crop_size = model.crop_size
        scale_size = model.scale_size
        input_mean = model.input_mean
        input_std = model.input_std
        self.transform = torchvision.transforms.Compose([
                       GroupScale(int(scale_size)),
                       GroupCenterCrop(crop_size),
                       Stack(roll=False),
                       ToTorchFormatTensor(div=True),
                       GroupNormalize(input_mean, input_std),
                   ])

        # Load TSM model
        model = torch.nn.DataParallel(model, device_ids=1).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device)['state_dict'])
        self.model = model
        self.model.eval()

        # Frame samples to be selected in a clip
        self.action_names = ['explore', 'investigate']
        self.rgb_sample = [2, 6, 9, 13, 17, 20, 24, 28]  # [4, 12, 19, 26, 34, 41, 48, 56]

        self.arena_mask = cv2.imread(arena_mask_path)
        if self.arena_mask is None:
            print("Arena Mask not loaded: %s" % arena_mask_path)
            exit(0)

    def preprocess_frames(self, clip):

        sampled_clip = [clip[i] for i in self.rgb_sample]
        new_frames = list()

        for f in sampled_clip:
            h, w, _ = f.shape
            # Crop the frame
            cropped_frame = f[Y_CROP:h-Y_CROP, X_CROP:w-X_CROP]

            # Downsample by a factor of 4
            downsampled_frame = cv2.resize(cropped_frame,
                                           (0, 0), fx=0.25, fy=0.25)

            # Convert to PIL Image
            new_frames.append(Image.fromarray(cv2.cvtColor(downsampled_frame, cv2.COLOR_BGR2RGB)))

        return self.transform(new_frames)

    def detect(self, clip):
        processed_clip = self.preprocess_frames(clip)

        out = self.model(processed_clip)[0]

        score, y_pred = F.softmax(out, dim=0).max(dim=0)

        return self.action_names[y_pred], float(score)

    def detect_location(self, frame):
        """
        1. First Color mask the image to obtain the "Region of Interest" (RoI) which contains pigs
        2. The above step degrades the pig image visible in the RoI. Hence, use Canny edge detector to 
           detect boundaries of images and dilate the boundaries to obtain big blobs
        3. Use the blobs obtained as the final mask. Tune the params such that RoI contains false positives,
           but no false negatives
        """
        # Resize the frame
        h, w, _ = frame.shape
        scaled_w, scaled_h = w//3, h//3

        low, high = (103, 35, 133), (106, 255, 255)
        frame = cv2.resize(frame, (scaled_w, scaled_h))
        frame = cv2.bitwise_and(frame, cv2.resize(self.arena_mask, (scaled_w, scaled_h)))

        # Smoothen the image using Gaussian Blur
        img = cv2.GaussianBlur(frame, (7, 7), 2)

        # Convert to HSV space and color mask the image
        hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv_img, low, high)
        hsv_masked = cv2.bitwise_and(img, img, mask=mask)

        # Convert result to Grayscale
        gray = cv2.cvtColor(hsv_masked, cv2.COLOR_RGB2GRAY)

        # Use canny detector to detect edges and dilate
        canny = cv2.Canny(gray, 26, 75)
        kernel = np.ones((7, 7))
        dilated = cv2.dilate(canny, kernel, iterations=4)
        thresh = cv2.erode(dilated, kernel, iterations=1)

        # Detect Contours
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        sorted_cnts = sorted(cnts, key=lambda x: cv2.contourArea(x), reverse=True)
        M = cv2.moments(sorted_cnts[0])
        x_contour = int(M["m10"] / M["m00"])

        return "left" if x_contour < (scaled_w//2) else "right"
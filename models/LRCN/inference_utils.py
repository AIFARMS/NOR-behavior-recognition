import sys
sys.path.append('../..')

import os
import cv2
import torch
import pickle
import numpy as np
import torchvision.transforms as transforms

from utils import cat2labels
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder

from PIL import Image

X_CROP, Y_CROP = 350,30
CLIP_LEN = 60

class FeatureExtractor():
	
	def __init__(self, cnn_model, cnn_checkpoint):
		cnn_model.load_state_dict(torch.load(cnn_checkpoint))
		self.model = cnn_model
		self.model.eval()

		self.mean = [0.485, 0.456, 0.406]
		self.std = [0.229, 0.224, 0.225]

		self.transform = transforms.Compose([
						transforms.Resize([224, 224]),
						transforms.ToTensor(),
						transforms.Normalize(mean=self.mean, std=self.std)])

	def extract(self, processed_frames):
		"""
		Return extracted features
		"""
		return self.model(processed_frames)

	def preprocess_frames(self, frame_buffer):

		frame_stack = []
		for f in frame_buffer:
			h, w, _ = f.shape
			## Crop the frame
			cropped_frame = f[Y_CROP:h-Y_CROP, X_CROP:w-X_CROP]
			
			## Downsample by a factor of 4
			downsampled_frame = cv2.resize(cropped_frame, (0,0), fx=0.25, fy=0.25) 

			## Convert to PIL Image
			frame_stack.append(self.transform(Image.fromarray(cv2.cvtColor(downsampled_frame, cv2.COLOR_BGR2RGB))))
		return torch.stack(frame_stack, dim=0).unsqueeze(0)

	def load_frames(self, frames_dir):

		frame_stack = []
		for i in range(CLIP_LEN):
			frame = Image.open(os.path.join(frames_dir, '%d.jpg'%(i+1)))
			frame = self.transform(frame)
			frame_stack.append(frame)

		return torch.stack(frame_stack, dim=0).unsqueeze(0)

class ActionDetector():

	def __init__(self, rnn_model, rnn_checkpoint):
		rnn_model.load_state_dict(torch.load(rnn_checkpoint))
		self.model = rnn_model
		self.model.eval()
		
		action_names = ['explore', 'investigate']

		# convert labels -> category
		self.le = LabelEncoder()
		self.le.fit(action_names)

	def detect(self, cnn_embed_seq):
		
		with torch.no_grad():
			output = self.model(cnn_embed_seq)
			output = F.softmax(output)
			confidence, y_pred = output.topk(1, largest=True, sorted=True)

			return cat2labels(self.le, y_pred[0].tolist())[0], confidence[0].tolist()[0]

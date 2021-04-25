import os
import cv2
import PIL
import numpy as np

from PIL import Image
from keras.models import Model
from keras.utils.data_utils import get_file

from c3d import C3D

C3D_MEAN_PATH = 'https://github.com/adamcasson/c3d/releases/download/v0.1/c3d_mean.npy'
X_CROP, Y_CROP = 350, 30

CLIP_LEN = 30

class FeatureExtractor():
	
	def __init__(self, weights_path):
		"""
		Initialize C3D model
		"""

		c3d_model = C3D(weights_path)
		layer_name = 'fc6'
		self.model = Model(inputs=c3d_model.input, outputs=c3d_model.get_layer(layer_name).output)

	def extract(self, clip):
		"""
		Return extracted features
		"""
		return self.model.predict(clip)[0]

	def preprocess_clip_stream(self, frames):
		"""
		Process clips from cv2.VideoCapture()
		"""
		new_frames = []
		for f in frames:
			h, w, _ = f.shape
			## Crop the frame
			cropped_frame = f[Y_CROP:h-Y_CROP, X_CROP:w-X_CROP]

			## Downsample by a factor of 4
			downsampled_frame = cv2.resize(cropped_frame, (0,0), fx=0.25, fy=0.25) 

			## Convert to PIL Image
			new_frames.append(Image.fromarray(cv2.cvtColor(downsampled_frame, cv2.COLOR_BGR2RGB)))
		return self.preprocess_clip(new_frames)

	@staticmethod
	def preprocess_clip(frames):
		"""
		Resize and subtract mean from clip input
		"""

		# Reshape to 128x171
		reshape_frames = np.zeros((16, 128, 171, 3))

		if CLIP_LEN == 60:
			frame_ids = [1 + i*3 for i in range(16)] 
		else:
			frame_ids = [i*2 for i in range(15)] + [29] 

		for i, fid in enumerate(frame_ids):
			img = np.array(frames[fid].resize([171, 128], resample=PIL.Image.BICUBIC))
			reshape_frames[i, :, :, :] = img

		mean_path = get_file('c3d_mean.npy',
							 C3D_MEAN_PATH,
							 cache_subdir='models',
							 md5_hash='08a07d9761e76097985124d9e8b2fe34')

		# Subtract mean
		mean = np.load(mean_path)
		reshape_frames -= mean
		
		# Crop to 112x112
		reshape_frames = reshape_frames[:, 8:120, 30:142, :]
		
		# Add extra dimension for samples
		reshape_frames = np.expand_dims(reshape_frames, axis=0)

		return reshape_frames

	@staticmethod
	def load_clip(clip_path):
		"""
		Given a path to a clip folder, read in all the frames in the clip using PIL
		"""
		frames = []
		for i in range(CLIP_LEN):
			image = Image.open(os.path.join(clip_path, '%d.jpg'%(i+1)))
			frames.append(image)
		return frames

	def load_and_extract(self, clip_path):
		"""
		Extract C3D features for a given clip using the clip folder
		"""
		clip = self.load_clip(clip_path)
		processed_clip = self.preprocess_clip(clip)
		return self.extract(processed_clip)

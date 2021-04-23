import cv2
import torch
import torchvision
import torch.nn.functional as F

from PIL import Image
from ops.models import TSN
from ops.transforms import *

X_CROP, Y_CROP = 350, 30

class TSM_detector():

	def __init__(self, modality, checkpoint):

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

		## Define transforms
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

		## Load TSM model
		model = torch.nn.DataParallel(model, device_ids=1).to(device)
		model.load_state_dict(torch.load(checkpoint)['state_dict'])
		self.model = model
		self.model.eval()

		self.action_names = ['explore', 'investigate']
		self.rgb_sample = [4, 12, 19, 26, 34, 41, 48, 56]

	def preprocess_frames(self, clip):

		sampled_clip = [clip[i] for i in self.rgb_sample]
		new_frames = list()

		for f in sampled_clip:
			h, w, _ = f.shape
			## Crop the frame
			cropped_frame = f[Y_CROP:h-Y_CROP, X_CROP:w-X_CROP]

			## Downsample by a factor of 4
			downsampled_frame = cv2.resize(cropped_frame, (0,0), fx=0.25, fy=0.25) 

			## Convert to PIL Image
			new_frames.append(Image.fromarray(cv2.cvtColor(downsampled_frame, cv2.COLOR_BGR2RGB)))

		return self.transform(new_frames)

	def detect(self, clip):
		processed_clip = self.preprocess_frames(clip)

		out = self.model(processed_clip)[0]

		score, y_pred = F.softmax(out, dim=0).max(dim=0)
		
		return self.action_names[y_pred], float(score)



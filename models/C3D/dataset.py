import os
import torch
import numpy as np
from torch.utils import data


class Dataset_C3D(data.Dataset):
	
	def __init__(self, data_path, np_feature_files, labels):
		self.data_path = data_path
		self.labels = labels
		self.np_feature_files = np_feature_files

	def __len__(self):
		return len(self.np_feature_files)

	def __getitem__(self, index):

		X = np.load(os.path.join(self.data_path, self.np_feature_files[index]))
		y = torch.LongTensor([self.labels[index]]) # (labels) LongTensor are for int64 instead of FloatTensor

		return X, y

"""
Binary Classifier on top of C3D features to distinguish between 'explore' and 'investigate'
"""

import torch
import torch.nn as nn

class BinaryClassifier(nn.Module):
	def __init__(self, checkpoint=None):
		super(BinaryClassifier, self).__init__()
		self.fc1 = nn.Linear(4096, 1024)
		self.fc2 = nn.Linear(1024, 256)
		self.fc3 = nn.Linear(256, 2)
		self.relu = nn.ReLU()
						   
	def forward(self,x):
		out = self.fc1(x)
		out = self.relu(out)
		out = self.fc2(out)
		out = self.relu(out)
		out = self.fc3(out)
		return out

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ResnetEncoder(nn.Module):
	def __init__(self, fc1_, fc2_, dropout, CNN_out):
		## Load the pretrained ResNet-X and replace top fc layer.
		super(ResnetEncoder, self).__init__()

		self.dropout = dropout

		resnet = models.resnet18(pretrained=True)
		modules = list(resnet.children())[:-1]      # delete the last fc layer.
		self.resnet = nn.Sequential(*modules)
		self.fc1 = nn.Linear(resnet.fc.in_features, fc1_)
		self.bn1 = nn.BatchNorm1d(fc1_, momentum=0.01)
		self.fc2 = nn.Linear(fc1_, fc2_)
		self.bn2 = nn.BatchNorm1d(fc2_, momentum=0.01)
		self.fc3 = nn.Linear(fc2_, CNN_out)
		
	def forward(self, x_3d):
		cnn_embed_seq = []
		## Iterate through all the frames in a clip
		for t in range(x_3d.size(1)):
			# ResNet CNN
			with torch.no_grad():
				x = self.resnet(x_3d[:, t, :, :, :])  # ResNet
				x = x.view(x.size(0), -1)             # flatten output of conv

			# FC layers
			x = self.bn1(self.fc1(x))
			x = F.relu(x)
			x = self.bn2(self.fc2(x))
			x = F.relu(x)
			x = F.dropout(x, p=self.dropout, training=self.training)
			x = self.fc3(x)

			cnn_embed_seq.append(x)

		# swap time and sample dim such that (sample dim, time dim, CNN latent dim)
		cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
		# cnn_embed_seq: shape=(batch, time_step, input_size)

		return cnn_embed_seq

class DecoderRNN(nn.Module):
	def __init__(self, CNN_out, h_RNN_layers, h_RNN, h_FC_dim, dropout, num_classes):
		super(DecoderRNN, self).__init__()

		self.RNN_input_size = CNN_out
		self.h_RNN = h_RNN                 # RNN hidden nodes
		self.h_FC_dim = h_FC_dim
		self.dropout = dropout
		self.num_classes = num_classes

		self.LSTM = nn.LSTM(
			input_size=self.RNN_input_size,
			hidden_size=self.h_RNN,        
			num_layers=h_RNN_layers,       
			batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
		)

		self.fc1 = nn.Linear(self.h_RNN, self.h_FC_dim)
		self.fc2 = nn.Linear(self.h_FC_dim, self.num_classes)

	def forward(self, x_RNN):
		
		self.LSTM.flatten_parameters()
		RNN_out, (h_n, h_c) = self.LSTM(x_RNN, None)  
		""" h_n shape (n_layers, batch, hidden_size), h_c shape (n_layers, batch, hidden_size) """ 
		""" None represents zero initial hidden state. RNN_out has shape=(batch, time_step, output_size) """

		# FC layers
		x = self.fc1(RNN_out[:, -1, :])   # choose RNN_out at the last time step
		x = F.relu(x)
		x = F.dropout(x, p=self.dropout, training=self.training)
		x = self.fc2(x)

		return x

class ResnetEncoderPi(nn.Module):
	def __init__(self, fc1_, fc2_, dropout, CNN_out):
		## Load the pretrained ResNet-X and replace top fc layer.
		super(ResnetEncoder, self).__init__()

		self.dropout = dropout

		resnet = models.resnet18(pretrained=True)
		modules = list(resnet.children())[:-1]      # delete the last fc layer.
		self.resnet = nn.Sequential(*modules)
		self.fc1 = nn.Linear(resnet.fc.in_features, fc1_)
		self.bn1 = nn.BatchNorm1d(fc1_, momentum=0.01)
		self.fc2 = nn.Linear(fc1_, fc2_)
		self.bn2 = nn.BatchNorm1d(fc2_, momentum=0.01)
		self.fc3 = nn.Linear(fc2_, CNN_out)
		
	def forward(self, x_2d):
		cnn_embed_seq = []
		
		with torch.no_grad():
			x = self.resnet(x_2d)  # ResNet
			x = x.view(x.size(0), -1)             # flatten output of conv

		# FC layers
		x = self.bn1(self.fc1(x))
		x = F.relu(x)
		x = self.bn2(self.fc2(x))
		x = F.relu(x)
		x = F.dropout(x, p=self.dropout, training=self.training)
		x = self.fc3(x)

		return x
		

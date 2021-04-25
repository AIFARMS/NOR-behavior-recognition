import os
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from utils import labels2cat, create_directory
from dataset import Dataset_CRNN 
from model import ResnetEncoder, DecoderRNN

ROOT_DIR = "../../data/compressed_action_frames-60"    
CHECKPOINT_DIR = "checkpoints/"
create_directory(CHECKPOINT_DIR)

# EncoderCNN architecture
CNN_FC1, CNN_FC2 = 1024, 1024
CNN_OUT = 512   
RESNET_INPUT_SIZE = 224        
DROPOUT = 0.0       # dropout probability

# DecoderRNN architecture
RNN_HIDDEN_LAYERS = 3
RNN_HIDDEN_NODES = 64
RNN_FC = 16

# training parameters
epochs = 30       # training epochs
batch_size = 128
LEARNING_RATE = 1e-4
log_interval = 1   # interval for displaying training info

def train(log_interval, model, device, train_loader, optimizer, epoch):    
	
	## Set model to training mode
	cnn_encoder, rnn_decoder = model
	cnn_encoder.train()
	rnn_decoder.train()

	losses, scores = [], []
	total_samples = 0   #counting total trained sample in one epoch
	
	for batch_idx, (X, y) in enumerate(train_loader):
		# distribute data to device
		X, y = X.to(device), y.to(device).view(-1, )        
		total_samples += X.size(0)
                
		optimizer.zero_grad()
		output = rnn_decoder(cnn_encoder(X))   # output = (batch size, number of classes)

		loss = F.cross_entropy(output, y)
		losses.append(loss.item())

		# to compute accuracy
		y_pred = torch.max(output, 1)[1]  # y_pred != output
	
		step_score = accuracy_score(y.cpu().data.squeeze().numpy(), y_pred.cpu().data.squeeze().numpy())
		scores.append(step_score)         # computed on CPU

		loss.backward()
		optimizer.step()

		# show information
		if (batch_idx + 1) % log_interval == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accu: {:.2f}%'.format(
				epoch + 1, total_samples, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), loss.item(), 100 * step_score))

	return losses, scores


def validation(model, device, optimizer, test_loader):
	
	## Set model to testing mode
	cnn_encoder, rnn_decoder = model
	cnn_encoder.eval()
	rnn_decoder.eval()

	all_y, all_y_pred = [], []
	test_loss = 0

	with torch.no_grad():
		for X, y in test_loader:
			# distribute data to device
			X, y = X.to(device), y.to(device).view(-1, )

			output = rnn_decoder(cnn_encoder(X))

			loss = F.cross_entropy(output, y, reduction='sum')
			test_loss += loss.item()                 # sum up batch loss
			y_pred = output.max(1, keepdim=True)[1]  # (y_pred != output) get the index of the max log-probability

			# collect all y and y_pred in all batches
			all_y.extend(y)
			all_y_pred.extend(y_pred)

	test_loss /= len(test_loader.dataset)

	# compute accuracy
	all_y = torch.stack(all_y, dim=0)
	all_y_pred = torch.stack(all_y_pred, dim=0)
	y_true, y_pred = all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy()
	test_score = accuracy_score(y_true, y_pred)
	precision, recall, threshold = precision_recall_curve(y_true, y_pred)
	print(precision, recall, test_score, average_precision_score(y_true, y_pred))

	# show information
	print('\nTest set ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(len(all_y), test_loss, 100* test_score))

	# save Pytorch models of best record
	torch.save(cnn_encoder.state_dict(), os.path.join(CHECKPOINT_DIR, 'cnn_encoder_epoch{}.pth'.format(epoch + 1)))  # save spatial_encoder
	torch.save(rnn_decoder.state_dict(), os.path.join(CHECKPOINT_DIR, 'rnn_decoder_epoch{}.pth'.format(epoch + 1)))  # save motion_encoder
	torch.save(optimizer.state_dict(), os.path.join(CHECKPOINT_DIR, 'optimizer_epoch{}.pth'.format(epoch + 1)))      # save optimizer
	print("Epoch {} model saved!".format(epoch + 1))

	return test_loss, test_score

def get_data(root_dir, label_encoder):
	fnames = os.listdir(root_dir)

	all_names, actions = [], []
	for f in fnames:
		loc1 = f.find('>')
		loc2 = f.find('-')

		actions.append(f[(loc1 + 1): loc2])
		all_names.append(f)

	# list all data files
	all_X_list = all_names                  
	all_y_list = labels2cat(label_encoder, actions)    

	return train_test_split(all_X_list, all_y_list, test_size=0.25, random_state=42)


if __name__ == '__main__':

	## Detect devices
	use_cuda = torch.cuda.is_available()                   
	device = torch.device("cuda" if use_cuda else "cpu")   
	print("Using %s"%("GPU" if use_cuda else "CPU"))

	## Data loading parameters
	params = {
		'batch_size': batch_size, 
		'shuffle': True, 
		'num_workers': 4, 
		'pin_memory': True
	} if use_cuda else {
		'batch_size': batch_size, 
		'shuffle': True, 
		'num_workers':28
	}


	## Encode labels using One Hot Encoding
	action_names = ['explore', 'investigate']

	label_encoder = LabelEncoder()
	label_encoder.fit(action_names)

	print("List of classes: ", list(label_encoder.classes_))

	action_category = label_encoder.transform(action_names).reshape(-1, 1)
	ohe = OneHotEncoder()
	ohe.fit(action_category)

	## Prepare Training and Testing dataset
	train_list, test_list, train_label, test_label = get_data(ROOT_DIR, label_encoder)

	transform = transforms.Compose([transforms.Resize([RESNET_INPUT_SIZE, RESNET_INPUT_SIZE]),
									transforms.ToTensor(),
									transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

	train_set = Dataset_CRNN(ROOT_DIR, train_list, train_label, transform=transform)
	valid_set =	Dataset_CRNN(ROOT_DIR, test_list, test_label, transform=transform)

	train_loader = data.DataLoader(train_set, **params)
	valid_loader = data.DataLoader(valid_set, **params)


	## Initialize model
	cnn_encoder = ResnetEncoder(fc1_=CNN_FC1, fc2_=CNN_FC2, dropout=DROPOUT, CNN_out=CNN_OUT).to(device)
	rnn_decoder = DecoderRNN(CNN_out=CNN_OUT, h_RNN_layers=RNN_HIDDEN_LAYERS, h_RNN=RNN_HIDDEN_NODES, 
							 h_FC_dim=RNN_FC, dropout=DROPOUT, num_classes=2).to(device)

	# Parallelize model to multiple GPUs
	if torch.cuda.device_count() > 1:
		print("Using", torch.cuda.device_count(), "GPUs!")
		cnn_encoder = nn.DataParallel(cnn_encoder)
		rnn_decoder = nn.DataParallel(rnn_decoder)

		# Combine all EncoderCNN + DecoderRNN parameters
		crnn_params = list(cnn_encoder.module.fc1.parameters()) + list(cnn_encoder.module.bn1.parameters()) + \
					  list(cnn_encoder.module.fc2.parameters()) + list(cnn_encoder.module.bn2.parameters()) + \
					  list(cnn_encoder.module.fc3.parameters()) + list(rnn_decoder.parameters())

	elif torch.cuda.device_count() == 1:
		print("Using", torch.cuda.device_count(), "GPU!")
		# Combine all EncoderCNN + DecoderRNN parameters
		crnn_params = list(cnn_encoder.fc1.parameters()) + list(cnn_encoder.bn1.parameters()) + \
					  list(cnn_encoder.fc2.parameters()) + list(cnn_encoder.bn2.parameters()) + \
					  list(cnn_encoder.fc3.parameters()) + list(rnn_decoder.parameters())
	
	else:
		crnn_params = list(cnn_encoder.fc1.parameters()) + list(cnn_encoder.bn1.parameters()) + \
					  list(cnn_encoder.fc2.parameters()) + list(cnn_encoder.bn2.parameters()) + \
					  list(cnn_encoder.fc3.parameters()) + list(rnn_decoder.parameters())

	optimizer = torch.optim.Adam(crnn_params, lr=LEARNING_RATE)

	validate = True
	if validate:
		cnn_encoder.load_state_dict(torch.load('checkpoints/cnn-pig.pth'))
		rnn_decoder.load_state_dict(torch.load('checkpoints/rnn-pig.pth'))
		validation([cnn_encoder, rnn_decoder], device, optimizer, valid_loader)
		exit()

	# record training process
	epoch_train_losses = []
	epoch_train_scores = []
	epoch_test_losses = []
	epoch_test_scores = []

	# start training
	import time
	for epoch in range(epochs):
		# train, test model
		print("Training epoch %d"%epoch)
		t = time.time()
		train_losses, train_scores = train(log_interval, [cnn_encoder, rnn_decoder], device, train_loader, optimizer, epoch)
		print("Training time: %f, Validating epoch %d"%(time.time()-t, epoch)); t = time.time()
		epoch_test_loss, epoch_test_score = validation([cnn_encoder, rnn_decoder], device, optimizer, valid_loader)
		print("Validation time: %f"%(time.time()-t))
		# save results
		epoch_train_losses.append(train_losses)
		epoch_train_scores.append(train_scores)
		epoch_test_losses.append(epoch_test_loss)
		epoch_test_scores.append(epoch_test_score)

		# save all train test results
		A = np.array(epoch_train_losses)
		B = np.array(epoch_train_scores)
		C = np.array(epoch_test_losses)
		D = np.array(epoch_test_scores)
		np.save('./CRNN_epoch_training_losses.npy', A)
		np.save('./CRNN_epoch_training_scores.npy', B)
		np.save('./CRNN_epoch_test_loss.npy', C)
		np.save('./CRNN_epoch_test_score.npy', D)

	# plot
	fig = plt.figure(figsize=(10, 4))
	plt.subplot(121)
	plt.plot(np.arange(1, epochs + 1), A[:, -1])  # train loss (on epoch end)
	plt.plot(np.arange(1, epochs + 1), C)         #  test loss (on epoch end)
	plt.title("model loss")
	plt.xlabel('epochs')
	plt.ylabel('loss')
	plt.legend(['train', 'test'], loc="upper left")
	# 2nd figure
	plt.subplot(122)
	plt.plot(np.arange(1, epochs + 1), B[:, -1])  # train accuracy (on epoch end)
	plt.plot(np.arange(1, epochs + 1), D)         #  test accuracy (on epoch end)
	plt.title("training scores")
	plt.xlabel('epochs')
	plt.ylabel('accuracy')
	plt.legend(['train', 'test'], loc="upper left")
	title = "./fig_ResNetCRNN.png"
	plt.savefig(title, dpi=600)
	# plt.close(fig)
	plt.show()

import os
import cv2
import torch

from model import ResnetEncoder, DecoderRNN
from inference_utils import ActionDetector, FeatureExtractor

## Load CNN encoder
cnn_encoder = ResnetEncoder(fc1_=1024, fc2_=1024, dropout=0.0, CNN_out=512)
extractor = FeatureExtractor(cnn_encoder, 'checkpoints/cnn-pig.pth')

## Load RNN decoder
rnn_decoder = DecoderRNN(CNN_out=512, h_RNN_layers=3, h_RNN=64, h_FC_dim=16, dropout=0, num_classes=2)
action_detector = ActionDetector(rnn_decoder, 'checkpoints/rnn-pig.pth')

ROOT_DIR = '../../data/compressed_action_frames-60/'

with torch.no_grad():

	for test_dir in os.listdir(ROOT_DIR): 
		processed_frames = extractor.load_frames(os.path.join(ROOT_DIR, test_dir))
		prediction, score = action_detector.detect(extractor.extract(processed_frames))

		print(test_dir, prediction, score)





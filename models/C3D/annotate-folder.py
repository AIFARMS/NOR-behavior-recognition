import os
import cv2
import torch

from model import BinaryClassifier
from c3d_detector import C3D_detector
from feature_extractor import FeatureExtractor

LOG = True
CLIP_LEN = 60

## Load C3D feature extractor
extractor = FeatureExtractor('c3d_sports1m.h5')

## Load Binary Classifier
classifier = BinaryClassifier()
detector = C3D_detector(classifier, checkpoint='checkpoints/model_epoch19.pth')

## Obtain list of videos
BASE_DIR = '../../data/videos/'
video_list = os.listdir(BASE_DIR)
print("Processing %d videos"%len(video_list))

for video_p in video_list:
	print(video_p)

	## Setup reading from video stream
	video_path = os.path.join(BASE_DIR, video_p)
	video_stream = cv2.VideoCapture(video_path)

	## Setup output video
	if not LOG:
		fourcc = cv2.VideoWriter_fourcc(*'mp4v')
		video_name = video_path.split('/')[-1].split('.')[0]
		out = cv2.VideoWriter(video_name+'-c3d.mp4', fourcc, 30, (1920, 1080))

	frame_id, frame_buffer = 0, []
	with torch.no_grad():
		while True:
			ret, frame = video_stream.read()
			if frame is None: break
			height, width, _ = frame.shape
			frame_id+=1

			## Buffer to annotate the original video
			frame_buffer.append(frame)

			## Collect and predict action for CLIP_LEN frames
			if len(frame_buffer) == CLIP_LEN:				
				processed_frames = extractor.preprocess_clip_stream(frame_buffer)
				prediction, score = detector.detect(extractor.extract(processed_frames))

				## Annotate CLIP_LEN frames and write it out to the output file
				for f in frame_buffer:
					cv2.putText(f,"%s: %f"%(prediction, score),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
					if not LOG:
						out.write(f)

				if LOG:
					print("%d %s %f"%(frame_id, prediction, score))
				## Reset the buffer
				frame_buffer = []

			## If the 'q' key is pressed, break from the loop
			key = cv2.waitKey(1) & 0xFF
			if key == ord("q"):
				break

	if not LOG:
		out.release()
	cv2.destroyAllWindows()

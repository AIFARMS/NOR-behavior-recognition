import os
import cv2
import torch

from detector import TSM_detector

LOG = True

## Load TSM detector
modality = "RGB"
detector = TSM_detector(modality, 'checkpoint/%s/ckpt.best.pth.tar'%modality.lower())

## Obtain list of videos
base_dir = '../../Livestock-Action-Recognition/data/videos/'
video_list = os.listdir(base_dir)
print("Processing %d videos"%len(video_list))

for video_p in video_list:
	print(video_p)

	## Setup reading from video stream
	video_path = os.path.join(base_dir, video_p)
	video_stream = cv2.VideoCapture(video_path)

	## Setup output video
	if not LOG:
		fourcc = cv2.VideoWriter_fourcc(*'mp4v')
		video_name = video_path.split('/')[-1].split('.')[0]
		out = cv2.VideoWriter(video_name+'-tsm-%s.mp4'%modality, fourcc, 30, (1920, 1080))

	frame_id, frame_buffer = 0, []
	with torch.no_grad():
		while True:
			ret, frame = video_stream.read()
			if frame is None: break
			height, width, _ = frame.shape
			frame_id+=1

			## Buffer to annotate the original video
			frame_buffer.append(frame)

			## Collect and predict action for 60 frames
			if len(frame_buffer) == 60:				
				prediction, score = detector.detect(frame_buffer)

				## Annotate 60 frames and write it out to the output file
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

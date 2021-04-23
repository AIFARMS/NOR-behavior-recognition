import cv2
import argparse

parser = argparse.ArgumentParser(description='Fix FPS of annotated videos')
parser.add_argument('-v', '--video_path', help='path to video', required=True, type=str)
parser.add_argument('-f', '--fps', help='FPS required', required=True, type=int)
args = parser.parse_args()

## Setup reading from video stream
video_stream = cv2.VideoCapture(args.video_path)

## Setup output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_name = args.video_path.split('/')[-1].split('.')[0]
out = cv2.VideoWriter(video_name+'-fixed.mp4', fourcc, args.fps, (1920, 1080))

while True:
	ret, frame = video_stream.read()
	if frame is None: break
	out.write(frame)

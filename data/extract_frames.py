import os
import cv2
import pandas as pd

ANNOTATIONS_DF = pd.read_csv('action_annotation.csv')
VIDEO_DIR = './videos'
X_CROP, Y_CROP = 350,30

def create_dir(dir_name):
	try: 
		os.stat(dir_name)
	except:
		os.mkdir(dir_name)

def crop_and_downsample(frame):
	h, w, _ = frame.shape

	## Crop the frame
	cropped_frame = frame[Y_CROP:h-Y_CROP, X_CROP:w-X_CROP]
	## Downsample by a factor of 4
	downsampled_frame = cv2.resize(cropped_frame, (0,0), fx=0.25, fy=0.25) 

	return downsampled_frame

## Create directory to store all the frames
FRAMES_DIR = 'compressed_action_frames'
create_dir(FRAMES_DIR)

video_files = os.listdir(VIDEO_DIR)
## Go over all the videos in the folder
for vf in video_files:
	video_name = vf.split('/')[-1].split('.')[0]
	video_stream = cv2.VideoCapture('%s/%s.mp4'%(VIDEO_DIR, video_name))

	## Create directory to store all frames per video
	print(video_name)
	# create_dir(os.path.join(FRAMES_DIR, video_name))

	actions_df = ANNOTATIONS_DF[ANNOTATIONS_DF['Scoring'] == video_name]

	## Skip initial frames	
	assert actions_df.iloc[0]['Behaviour'] == 'Begin Trial'

	start_frame, end_frame = int(actions_df.iloc[0]['Start_Frame']), int(actions_df.iloc[0]['Start_Frame'])
	[video_stream.read() for _ in range(start_frame)]
	actions_df = actions_df.iloc[1:]

	## Start annotating actions
	explore_count, investigate_count = 0, 0
	for index, row in actions_df.iterrows():

		action = row['Behaviour']
		astart_frame = int(row['Start_Frame']) 
		## Annotate exploration
		if end_frame < astart_frame:
			dir_path = os.path.join(FRAMES_DIR, '%s>explore-%d[%d-%d]'%(video_name, explore_count, end_frame, astart_frame))
			explore_count += 1
			create_dir(dir_path)

			frame_id = 1
			for i in range(astart_frame - end_frame):
				_, frame = video_stream.read()
				cv2.imwrite('%s/%d.jpg'%(dir_path,frame_id), crop_and_downsample(frame))
				frame_id += 1

		if action == 'End Trial':
			break
		aend_frame = int(row['Stop_Frame'])

		action = action.split(' ')[0].lower()
		## Annotate investigation
		dir_path = os.path.join(FRAMES_DIR, '%s>investigate-%s-%d[%d-%d]'%(video_name, action, investigate_count,astart_frame,aend_frame))
		investigate_count += 1
		create_dir(dir_path)

		frame_id = 1
		for i in range(aend_frame - astart_frame):
			_, frame = video_stream.read()
			cv2.imwrite('%s/%d.jpg'%(dir_path,frame_id), crop_and_downsample(frame))
			frame_id += 1

		start_frame, end_frame = astart_frame, aend_frame


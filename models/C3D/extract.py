import os
import numpy as np
from feature_extractor import FeatureExtractor

def create_dir(dir_name):
	try: 
		os.stat(dir_name)
	except:
		os.mkdir(dir_name)

ROOT_DIR = '../data/compressed_action_frames-60-all'
FEATURES_DIR = '../data/c3d_features'
create_dir(FEATURES_DIR)

extractor = FeatureExtractor('c3d_sports1m.h5')

for clip_name in os.listdir(ROOT_DIR):
	print(clip_name)
	clip_path = os.path.join(ROOT_DIR, clip_name)
	features = extractor.load_and_extract(clip_path)
	np.save(os.path.join(FEATURES_DIR, clip_name+'.npy'), features)

	

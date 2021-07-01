"""
Contains code to annotate all the videos in a specific folder using TSM
"""

import os
import cv2
import json
import math
import torch
import pickle
import argparse

from detector import TSM_detector

# Number of frames to buffer
BUFFER_SIZE = 30

fps_120 = ['548', '694', '616']

def _generate_metrics(investigation_times, novel_location, start_frame=0, fps=30.):

    # print(investigation_times)
    n = len(investigation_times["left"]) + len(investigation_times["right"])
    cd = (sum(i for _, i in investigation_times["left"]) +
          sum(i for _, i in investigation_times["right"])) / fps
    me = 1. * cd / n

    if investigation_times["left"]:
        _lf = investigation_times["left"][0][0]
        _ll = investigation_times["left"][-1][0]
    else:
        _lf = math.inf
        _ll = fps*300 ## End of the experiment

    if investigation_times["right"]:
        _lf = min(_lf, investigation_times["right"][0][0])
        _ll = max(_ll, investigation_times["right"][-1][0])

    lf = (_lf - start_frame) / fps
    ll = (_ll - start_frame) / fps

    RI = sum(i for _, i in investigation_times[novel_location]) / (cd * fps)

    return {
        "n" : n,
        "cd": cd,
        "me": me,
        "lf": lf,
        "ll": ll,
        "RI": RI
    }

def generate_metrics(annotation_dict, novel_location, start_frame=0, fps=30):
    """
    Generates the following metrics:
    n: total number of investigations
    cd: total time spent investigating
    me: average amount of time for any one investigation
    lf: latency to the first investigation
    ll: latency to the last investigation
    RI (recognition index): the amount of time spent investigating the novel
        object compared to the total amount of time investigating both objects
    """
    assert novel_location in ["left", "right"]

    sorted_keys = sorted([int(k) for k in annotation_dict["data"].keys()])
    annotation_list = [(annotation_dict["data"][str(k)]["action"], annotation_dict["data"][str(k)]["location"]) for k in sorted_keys]

    investigation_times = {
        "left"  : [],
        "right" : []
    }

    in_investigate, time_elapsed = None, 0
    for prediction, location in annotation_list:

        # If the current prediction is investigate
        if prediction == "investigate":

            # And if it was an ongoing investigation
            if in_investigate is not None:

                # And if the pig is at the same location, then simply increment
                if in_investigate == location:
                    investigation_times[location][-1][1] += BUFFER_SIZE
                # If the location has changed from left to right, then initiate another investigation
                else:
                    in_investigate = location
                    investigation_times[location].append([time_elapsed, BUFFER_SIZE])

            # Start an investigation
            else:
                in_investigate = location
                investigation_times[location].append([time_elapsed, BUFFER_SIZE])

            time_elapsed += BUFFER_SIZE

        # Set as "No investigation"
        else:
            in_investigate = None
            time_elapsed += BUFFER_SIZE

    return _generate_metrics(investigation_times, novel_location, start_frame, fps)


def generate_annotations(base_dir, video_path, json_dir='./'):

    video_name = video_path.split('/')[-1].split('.')[0]

    json_path = os.path.join(json_dir, "%s.json" % video_name)

    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            annotation_dict = json.load(f)

            annotation_dict["metrics"] = generate_metrics(annotation_dict,
                                                          annotation_dict["novel_location"], 
                                                          annotation_dict["start_frame"],
                                                          fps=120 if ('548' in video_path 
                                                            or '694' in video_path or '616' in video_path) else 30)
            # print(video_name, annotation_dict["metrics"])
            with open(json_path, "w") as f:
                json.dump(annotation_dict, f, indent=4)

    else:
        annotation_dict = {
            "video_name": video_name,
            "data": {}
        }

        # Setup reading from video stream
        video_path = os.path.join(base_dir, video_path)
        video_stream = cv2.VideoCapture(video_path)
        
        w, h, video_fps = int(video_stream.get(3)), int(video_stream.get(4)), video_stream.get(5)
        out = cv2.VideoWriter(f"{video_name}-prediction.mp4", cv2.VideoWriter_fourcc(*'mp4v'), video_fps, (w, h))

        frame_id, frame_buffer = 0, []
        with torch.no_grad():
            while True:
                ret, frame = video_stream.read()
                if frame is None:
                    break
                height, width, _ = frame.shape
                frame_id += 1

                # Buffer to annotate the original video
                frame_buffer.append(frame)

                # Collect and predict action for 30 frames
                if len(frame_buffer) == BUFFER_SIZE:
                    prediction, score = detector.detect(frame_buffer)

                    location = detector.detect_location(frame_buffer[-1].copy()) if prediction == "investigate" else "None"

                    print("%d %s %f %s" % (frame_id, prediction, score, location))
                    annotation_dict["data"][frame_id] = {
                        "action": prediction,
                        "confd": score,
                        "location": location
                    }

                    for f in frame_buffer:
                        cv2.putText(f,"%s %s: %f"%("" if location == "None" else location, prediction, score),
                            (30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
                        out.write(f)

                    # Reset the buffer
                    frame_buffer = []

        out.release()
        video_stream.release()

    return annotation_dict

if __name__ == '__main__':
# if __name__ == '__not_main__':

    parser = argparse.ArgumentParser(description="PNCL NOR Action Annotation")
    parser.add_argument('--video_path', '-v', required=True, help="Path to the video to be annotated")
    parser.add_argument('--checkpoint_path', '-c', required=True, help="Path to checkpoint")
    parser.add_argument('--mask_path', '-m', required=True, help="Path to Mask image")
    parser.add_argument('--json_dir', '-j', required=True, help="Directory where JSON files are saved")
    args = parser.parse_args()

    # Load TSM detector
    detector = TSM_detector("RGB", args.checkpoint_path, args.mask_path)

    # Annotate the video
    annotation_dict = generate_annotations('./', args.video_path, args.json_dir)

# if __name__ == '__main__':
if __name__ == '__not_main__':
    import pandas as pd

    parser = argparse.ArgumentParser(description="PNCL NOR Action Annotation")
    parser.add_argument('--video_path', '-v', required=True, help="Path to the video to be annotated")
    parser.add_argument('--checkpoint_path', '-c', required=True, help="Path to checkpoint")
    parser.add_argument('--mask_path', '-m', required=True, help="Path to Mask image")
    parser.add_argument('--json_dir', '-j', required=True, help="Directory where JSON files are saved")
    args = parser.parse_args()

    # Load TSM detector
    detector = TSM_detector("RGB", args.checkpoint_path, args.mask_path)

    # Obtain list of videos
    video_list = os.listdir(args.video_path)
    print("Processing %d videos" % len(video_list))
    metrics_df_dict = {
        'filename': [],
        'n': [],
        'cd': [],
        'me': [],
        'lf': [],
        'll': [],
        'RI': []
    }
    
    for video_p in video_list:
        annotation_dict = generate_annotations('./', os.path.join(args.video_path, video_p), args.json_dir)

        metrics_df_dict['filename'].append(video_p.split('.')[0])
        for k, val in annotation_dict["metrics"].items():
            metrics_df_dict[k].append(val)
    
    metrics_df = pd.DataFrame(metrics_df_dict)
    print(metrics_df)
    metrics_df.to_csv('analysis/ai_metrics.csv', index=False)
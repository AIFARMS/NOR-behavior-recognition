# NOR-behavior-recognition

Official Implementation of **Vision-based Behavioral Recognition of Novelty Preference in Pigs**

<img src="data/result.gif" width="800"></img>

## Dataset

Download the dataset and the annotations from [this](https://drive.google.com/drive/folders/14XUYxM15NAI-zBrntrmQofhLv5otAw5b?usp=sharing) drive link and place under the ``data`` folder. 
Use the script [extract_frames.py](data/extract_frames.py) to pre-process the annotated frames from the dataset.

## LRCN 

Run ``python3 train.py`` to train the model. Run ``python3 annotate.py`` to annotate the video dataset

## C3D

Precompute C3D features using the script [extract.py](C3D/extract.py). Run ``python3 train.py`` to train the model. Run ``python3 annotate-folder.py`` to annotate the video dataset.

## TSM

Follow the procedure specified [here](https://github.com/mit-han-lab/temporal-shift-module) to generate the dataset. Run the following command to train the model:
```
python3 main.py pig RGB \
      -p 2 --arch resnet18 --num_segments 8  --gd 20 --lr 0.02 \
      --wd 1e-4 --lr_steps 12 25 --epochs 35 --batch-size 64 -j 16 --dropout 0.5 \
      --consensus_type=avg --eval-freq=1 --shift --shift_div=8 --shift_place=blockres --npb
``` 
Run ``python3 annotate.py`` to annotate the video dataset.

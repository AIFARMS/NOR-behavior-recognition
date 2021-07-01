# NOR-behavior-recognition

Official Implementation of **Vision-based Behavioral Recognition of Novelty Preference in Pigs**

**Arxiv Paper: https://arxiv.org/abs/2106.12181 ([CVPR2021 CV4Animals Workshop Poster](poster.pdf))**

<img src="data/result.gif" width="800"></img>

## Dataset

Download the dataset and the annotations from [this](https://drive.google.com/drive/folders/14XUYxM15NAI-zBrntrmQofhLv5otAw5b?usp=sharing) drive link and place under the ``data`` folder. 
Use the script [extract_frames.py](data/extract_frames.py) to extract and downsample the annotated frames from the dataset.
Use [statistic.ipynb](data/statistic.ipynb) to truncate clips into a fixed length of either 30 or 60 frames.

## Demo
Create a directory ``checkpoints`` and place [this](https://drive.google.com/file/d/1760tltQeVDdfr45KCvvXRBgvN-UEHZqa/view?usp=sharing) TSM checkpoint in the ``checkpoints`` folder. Run ``annotate.py`` using the following sample command:

```
python3 annotate.py -v data/videos/1815_C2_624_4wk.mp4 -c checkpoints/tsm.best.pth.tar -m data/pncl-maskfilter.png -j data/
```

## LRCN 
Run ``python3 train.py`` to train the model. 

To use pretrained model, download the `cnn-pig.pth` and `rnn-pig.pth` from this [drive](https://drive.google.com/drive/folders/1xx6G0JmaLFX8umIK5iagGJWZa_qAVdn6?usp=sharing) and place in the `models/LRCN/checkpoints` folder

Run ``python3 annotate-folder.py`` to annotate the video dataset

## C3D
Download the C3D sports-1m weights using 
```
cd models/C3D
wget https://github.com/adamcasson/c3d/releases/download/v0.1/sports1M_weights_tf.h5 -o c3d_sports1m.h5
```
Precompute C3D features for the dataset using the script [extract.py](C3D/extract.py). 

Run ``python3 train.py`` to train the model. 

Run ``python3 annotate-folder.py`` to annotate the video dataset.

## TSM

Follow the procedure specified [here](https://github.com/mit-han-lab/temporal-shift-module) to generate the dataset. Run the following command to train the model:
```
python3 main.py pig RGB \
      -p 2 --arch resnet18 --num_segments 8  --gd 20 --lr 0.02 \
      --wd 1e-4 --lr_steps 12 25 --epochs 35 --batch-size 64 -j 16 --dropout 0.5 \
      --consensus_type=avg --eval-freq=1 --shift --shift_div=8 --shift_place=blockres --npb
``` 
Run ``python3 annotate.py`` to annotate the video dataset.

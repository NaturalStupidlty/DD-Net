# DDNet

This is a Ternsorflow(Keras) implementation of the DDNet – Double-feature Double-motion Network

****
### Architecture
The DDNet architecture is described in this paper:
https://arxiv.org/pdf/1907.09658.pdf
[![](assets/architecture.png)](https://arxiv.org/pdf/1907.09658.pdf)
****

### Results 
Results on [SHREC](http://www-rech.telecom-lille.fr/shrec2017-hand/) dataset (using 3D skeletons)
[![](assets/results.png)](https://arxiv.org/pdf/1907.09658.pdf)
****

### Usage

This repository contains sources directory with implementation.

`sources/DDNet.py` - contains the actual model class

`sources/create_dataset.py` - script for creating a binary files with skeletons and labels (with pickle) from a
dataset that has a structure like [SHREC](http://www-rech.telecom-lille.fr/shrec2017-hand/). Note that you may need to
modify it if the structure of your dataset is different.

```
+---gesture_1 
|   +---finger_1 
|  |   +---subject_1 
|  |  |   +---essai_1 
|  |  |   |   
|  |  |   |   depth_0.png 
|  |  |   |   depth_1.png 
|  |  |   |   ... 
|  |  |   |   depth_N-1.png 
|  |  |   |   general_informations.txt 
|  |  |   |   skeletons_image.txt 
|  |  |   |   skeletons_world.txt 
|  |  |   | 
|  |  |   \---essai_2 
|  |  |   ... 
|  |  |   \---essai_5 
|  |   \---subject_2 
|  |   ... 
|  |   \---subject_20 
|   \---finger_2 
... 
\---gesture_14 
train_gestures.txt
test_gestures.txt
display_sequence.m
display_sequence.py
```

`souces/utils.py` - contains an important Config class with configurations for model training. There are also all
utilisation functions for the model training.

`sources/train.py` - script for training a model on a preprocessed dataset like
[SHREC](http://www-rech.telecom-lille.fr/shrec2017-hand/). Note that can modify this script to work with your own 
dataset.
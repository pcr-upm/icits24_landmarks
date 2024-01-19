# Landmark Detection using ICITS (2024)

#### Requisites
- images_framework https://github.com/pcr-upm/images_framework
- scipy
- torch
- torchvision
- timm

#### Installation
This repository must be located inside the following directory:
```
images_framework
    └── alignment
        └── icits24_landmarks
```
#### Usage
```
usage: icits24_landmarks_test.py [-h] [--input-data INPUT_DATA] [--show-viewer] [--save-image]
```

* Use the --input-data option to set an image, directory, camera or video file as input.

* Use the --show-viewer option to show results visually.

* Use the --save-image option to save the processed images.
```
usage: Alignment --database DATABASE
```

* Use the --database option to select the database model.
```
usage: ICITS24Landmarks [--gpu GPU]
```

* Use the --gpu option to set the GPU identifier (negative value indicates CPU mode).
```
> python images_framework/alignment/icits24_landmarks/test/icits24_landmarks_test.py --input-data images_framework/alignment/icits24_landmarks/test/example.tif --database wflw --gpu 0 --save-image
```

# Static FER
author: HBen  
Edit Time：2023.01.05
### Structure
```
yolov8_DAN_fer
│   README.md
│   main.ipynb:testing process  
│   fer.py:Use capture of the computer get the image . then show the detection image 
|   config.json: project define
│
└───model
│   │   dan.py: single face FER model
│   │   YOLOV8.py:face Detection
│   
│   
└───weights
    │   epochs_18_87.451_DAN.pth
    │   rafdb_epoch21_acc0.897_bacc0.8275.pth
    │   resnet18_msceleb.pth
    │   yolov8n-face.onnx
```
### How to use
- environments:
  - python >= 3.7
  - opencv-python >=4.7.0.72
  - pytorch >= 1.12.0
- Run
```commandline
python fer.py
```

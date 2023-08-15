import cv2
from PIL import Image
import json
import torch
import numpy as np
from model.YOLOV8 import YOLOv8_face
from model.dan import FERModel
from torchvision import transforms

if __name__ == "__main__":
    parameterRead=None
    with open('config.json') as f:
        parameterRead = json.load(f)
    Yolodefine=parameterRead['Yolov8']
    Dandefine=parameterRead['Dan']
    device_id=parameterRead['DEVICE']
    DEVICE=torch.device(device_id)

    # Initialize YOLOv8_face object detector
    YOLOv8_face_detector = YOLOv8_face(Yolodefine['modelpath'], conf_thres=Yolodefine['confThreshold'], iou_thres=Yolodefine['nmsThreshold'])
    Fermodel=FERModel(Dandefine['modelpath'])


    cap = cv2.VideoCapture(0)
    while(True):
        # 一帧一帧捕捉
        ret, srcimg = cap.read()
        boxes, scores, classids, kpts = YOLOv8_face_detector.detect(srcimg)
        #crop_img=YOLOv8_face_detector.crop_detections(srcimg,boxes,scores,kpts)
        #fer="None"
        #if crop_img is not None:
        img=Image.fromarray(cv2.cvtColor(srcimg,cv2.COLOR_BGR2RGB))
        dstimg=Fermodel.detect(img,boxes)
        #  print(fer)
        # 显示返回的每帧
        #dstimg = YOLOv8_face_detector.draw_detections(srcimg, boxes,fer)
        cv2.imshow('frame',dstimg)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
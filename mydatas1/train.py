#coding:utf-8
# 替换主干网络，训练
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8-p2.yaml')
    #model.load('yolov8n') # loading pretrain weights
    model.train(data='555.yaml', epochs=400, batch=8,lr0=0.01,resume=True)


import xml.etree.ElementTree as ET
import pickle
import numpy as np
import os
from os import listdir, getcwd
from os.path import join
sets = ['train', 'test', 'val']
classes = ['craze','white mark','joint line','bump','off-centered']
#classes =  ['crazing','inclusion','patches', 'pitted_surface', 'scratches','rolled-in_scale']

def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    x = format(x, '.6f')
    w = w * dw
    w = format(w, '.6f')
    y = y * dh
    y = format(y, '.6f')
    h = h * dh
    h = format(h, '.6f')
    return (x, y, w, h)
def convert_annotation(image_id):
    in_file = open('/home/gzz/zhoulin/model-sound code/yolov8/mydatas/Annotations/%s.xml' % (image_id),encoding='utf-8')
    out_file = open('/home/gzz/zhoulin/model-sound code/yolov8/mydatas/labels/%s.txt' % (image_id), 'w',encoding='utf-8')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
wd = getcwd()
print(wd)
for image_set in sets:
    if not os.path.exists('/home/gzz/zhoulin/model-sound code/yolov8/mydatas/labels/'):
        os.makedirs('/home/gzz/zhoulin/model-sound code/yolov8/mydatas/labels/')
    image_ids = open('/home/gzz/zhoulin/model-sound code/yolov8/mydatas/ImageSets/%s.txt' % (image_set)).read().strip().split()
    list_file = open('/home/gzz/zhoulin/model-sound code/yolov8/mydatas/%s.txt' % (image_set), 'w')
    for image_id in image_ids:
        list_file.write('/home/gzz/zhoulin/model-sound code/yolov8/mydatas/images/%s.jpg\n' % (image_id))
        convert_annotation(image_id)
    list_file.close()

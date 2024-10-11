import os
import random
trainval_percent = 0.9
train_percent = 0.8
xmlfilepath = '/home/gzz/zhoulin/model-sound code/yolov8/mydatas/Annotations'
txtsavepath = '/home/gzz/zhoulin/model-sound code/yolov8/mydatas/images'
total_xml = os.listdir(xmlfilepath)
num = len(total_xml)
list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)
ftrainval = open('/home/gzz/zhoulin/model-sound code/yolov8/mydatas/ImageSets/trainval.txt', 'w')
ftest = open('/home/gzz/zhoulin/model-sound code/yolov8/mydatas/ImageSets/test.txt', 'w')
ftrain = open('/home/gzz/zhoulin/model-sound code/yolov8/mydatas/ImageSets/train.txt', 'w')
fval = open('/home/gzz/zhoulin/model-sound code/yolov8/mydatas/ImageSets/val.txt', 'w')
for i in list:
    name = total_xml[i][:-4] + '\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)
ftrainval.close()
ftrain.close()
fval.close()
ftest.close()
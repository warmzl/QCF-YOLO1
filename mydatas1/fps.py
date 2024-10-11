import os

from ultralytics import YOLO


def load_model(model_path):
    model = YOLO(model_path)
    print('查看当前模型：', model)
    return model


if __name__ == '__main__':
    imgs_path = 'test'  # TODO 验证集目录   要求图像数量 >= 200
    model = load_model('ultralytics/cfg/models/v8/runs/detect/c3ghost-c2f-fpn-p2/weights/best.pt' )  # TODO 模型路径
    re_num = 10  # TODO 预热图像张数
    detect_count = 200  # TODO 推理图像张数
    images = os.listdir(imgs_path)
    count = 0
    times = []
    for item in images:
        if count < re_num:
            model(imgs_path + os.sep + item)
        elif count < re_num + detect_count:
            results = model(imgs_path + os.sep + item)
            times.append(results[0].speed)
        else:
            break
        count += 1
    # main(opt)
    time_sum = 0.
    for i in times:
        time_sum += sum(i.values())
    one_img_time = time_sum / detect_count
    FPS = 1000 / one_img_time
    print(f'FPS: {FPS}')


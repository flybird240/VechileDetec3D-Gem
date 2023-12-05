from PyQt5.QtCore import pyqtSignal, QObject
import torch
import time
import torchvision
import numpy as np
import cv2 as cv
import os

class MaskDetectWork(QObject):
    send_detect_img = pyqtSignal(object)  # 传送检测后的图片信号

    def __init__(self):
        super(MaskDetectWork, self).__init__()
        namePath = os.path.join('./models/mask_rcnn_pretrained_models', 'coco_mask.names')  # name路径
        with open(namePath, 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')
        os.environ['TORCH_HOME'] = './models/mask_rcnn_pretrained_models'
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)  # torchvision中的mask rcnn模型
        self.model.eval()
        self.img = None
        self.flush = True

        # 使用GPU
        self.train_on_gpu = torch.cuda.is_available()
        if self.train_on_gpu:
            self.model.cuda()
        self.conf_threshold = 0.5
        self.mask_threshold = 0.5

    def work(self):
        self.flush = False  # 刷新标志位先设为Fasle，ontime先不刷新图像界面
        t1 = time.time()
        self.img = cv.cvtColor(self.img, cv.COLOR_BGR2RGB)
        self.tensor_img = torch.from_numpy(self.img / 255.).permute(2, 0, 1).float().cuda()  # 转为tensor格式
        c, h, w = self.tensor_img.shape
        output = self.model([self.tensor_img])[0]  # 加上batch_size维度
        boxes = output['boxes'].cpu().detach().numpy()
        scores = output['scores'].cpu().detach().numpy()
        labels = output['labels'].cpu().detach().numpy()
        masks = output['masks'].cpu().detach().numpy()

        index = 0
        color_mask = np.zeros((h, w, c), dtype=np.uint8)
        mv = cv.split(color_mask)
        for x1, y1, x2, y2 in boxes:
            # 此处添加是否是车辆类判别
            if labels[index] not in [3, 6, 7, 8]:
                continue
            if scores[index] > self.conf_threshold:
                cv.rectangle(self.img, (np.int32(x1), np.int32(y1)), (np.int32(x2), np.int32(y2)), (0, 0, 255), 2, 8, 0)
                mask = np.squeeze(masks[index] > self.mask_threshold)
                np.random.randint(0, 256)
                mv[2][mask == 1], mv[1][mask == 1], mv[0][mask == 1] = [np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)]
                label_id = labels[index]
                label_txt = self.classes[label_id] + str(": %.2f" % scores[index])
                cv.putText(self.img, label_txt, (np.int32(x1), np.int32(y1)), cv.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 2)
            index += 1
        color_mask = cv.merge(mv)
        detect_img = cv.addWeighted(self.img, 0.7, color_mask, 0.3, 0)
        detect_img = cv.cvtColor(detect_img, cv.COLOR_BGR2RGB)
        t2 = time.time()
        fps = 'FPS: %.2f' % (1. / (t2 - t1))
        cv.putText(detect_img, fps, (0, 40), cv.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 255), 2)

        self.send_detect_img.emit(detect_img)  # 检测完成传送信号，触发槽函数接收并显示检测结果
        self.flush = True  # 界面ontime可刷新界面图像




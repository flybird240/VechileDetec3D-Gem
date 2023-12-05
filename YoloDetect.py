import numpy as np
import cv2 as cv
import os
import time
from PyQt5.QtCore import pyqtSignal, QObject

class YoloDetectWork(QObject):
    send_detect_img = pyqtSignal(object)  # 设置传送处理后图像信号
    send_detect_result = pyqtSignal(object)  # 设置传送***信号

    def __init__(self):
        super(YoloDetectWork, self).__init__()
        yolo_dir = './models/yolov3'  # YOLO文件路径
        weightsPath = os.path.join(yolo_dir, 'yolov3.weights')  # 权重文件
        configPath = os.path.join(yolo_dir, 'yolov3.cfg')  # 配置文件
        labelsPath = os.path.join(yolo_dir, 'coco.names')  # label名称
        with open(labelsPath, 'rt') as f:
            self.labels = f.read().rstrip('\n').split('\n')
        self.net = cv.dnn.readNetFromDarknet(configPath, weightsPath)  # 加载网络、配置权重
        self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)  # 配置GPU环境
        self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
        self.img = None
        self.flush = True
        self.CONFIDENCE = 0.5  # 过滤弱检测的最小概率
        self.THRESHOLD = 0.4  # 非最大值抑制阈值

    def work(self):
        self.flush = False # 刷新标志位先设为Fasle，ontime先不刷新图像界面
        t1 = time.time()
        # net需要的输入是blob格式的，用blobFromImage这个函数来转格式
        blobImg = cv.dnn.blobFromImage(self.img, 1.0 / 255.0, (416, 416), None, True, False)
        self.net.setInput(blobImg)  # 调用setInput函数将图片送入输入层
        # 获取网络输出层信息（所有输出层的名字），设定并前向传播
        outInfo = self.net.getUnconnectedOutLayersNames()
        layerOutputs = self.net.forward(outInfo)  # 得到各个输出层的、各个检测框等信息，是二维结构

        (H, W) = self.img.shape[:2]  # 拿到图片尺寸
        # 过滤layerOutputs
        # layerOutputs的第1维的元素内容: [center_x, center_y, width, height, objectness, N-class score data]
        # 过滤后的结果放入：
        boxes = []  # 所有边界框（各层结果放一起）
        confidences = []  # 所有置信度
        classIDs = []  # 所有分类ID

        # # 1）过滤掉置信度低的框框
        for out in layerOutputs:  # 各个输出层
            for detection in out:  # 各个框框
                # 拿到置信度
                scores = detection[5:]  # 各个类别的置信度
                classID = np.argmax(scores)  # 最高置信度的id即为分类id
                # 此处添加是否是车辆类判别
                if classID not in [2, 5, 6, 7]:
                    continue
                confidence = scores[classID]  # 拿到置信度

                # 根据置信度筛查
                if confidence > self.CONFIDENCE:
                    box = detection[0:4] * np.array([W, H, W, H])  # 将边界框放会图片尺寸
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # # 2）应用非最大值抑制(non-maxima suppression，nms)进一步筛掉
        idxs = cv.dnn.NMSBoxes(boxes, confidences, self.CONFIDENCE, self.THRESHOLD)  # boxes中，保留的box的索引index存入idxs
        # 应用检测结果
        np.random.seed(42)
        COLORS = np.random.randint(0, 255, size=(len(self.labels), 3), dtype="uint8")  # 框框显示颜色，每一类有不同的颜色，每种颜色都是由RGB三个值组成的，所以size为(len(labels), 3)
        if len(idxs) > 0:
            for i in idxs.flatten():  # indxs是二维的，第0维是输出层，所以这里把它展平成1维
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                color = [int(c) for c in COLORS[classIDs[i]]]
                cv.rectangle(self.img, (x, y), (x + w, y + h), color, 2)  # 线条粗细为2px
                text = "{}: {:.4f}".format(self.labels[classIDs[i]], confidences[i])
                cv.putText(self.img, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)  # cv.FONT_HERSHEY_SIMPLEX字体风格、0.5字体大小、粗细2px

        t2 = time.time()
        fps = 'FPS: %.2f' % (1. / (t2 - t1))
        cv.putText(self.img, fps, (0, 40), cv.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 255), 2)
        self.send_detect_img.emit(self.img)  # 信号传回主界面图像显示区
        self.flush = True # 界面ontime可刷新界面图像


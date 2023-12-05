# Created by: ww 2020/8/11
#界面与逻辑分离，主窗口逻辑代码

import os
import cv2 as cv
import sys
from MainDetect_UI import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QMainWindow
from YoloDetect import YoloDetectWork
from MaskRcnnDetect import MaskDetectWork


class Main(QMainWindow, Ui_MainWindow):

    startThreadSignal = pyqtSignal()  # 触发子线程信号
    """重写主窗体类"""
    def __init__(self, parent=None):
        super(Main, self).__init__(parent)
        self.setupUi(self) # 初始化窗体显示
        self.timer = QTimer(self) # 初始化定时器
        # 设置在label中自适应显示图片
        self.label_PrePicShow.setStyleSheet("QLabel{background-color:rgb(0,0,0);}")
        self.label_AftPicShow.setStyleSheet("QLabel{background-color:rgb(0,0,0);}")
        self.frame = None

        # 设置多线程检测
        self.thread = QThread()
        self.destroyed.connect(self.destroy_thread) # 关掉窗口结束进程

    def onbuttonclick_selectDataType(self, index):
        """选择输入数据类型(图像,视频)"""
        if index == 1:
            filename, _ = QFileDialog.getOpenFileName(self, "选择图像", os.getcwd(), "Images (*.jpg *.png);;All (*)")
            self.frame = cv.imread(filename)
            self.label_PrePicShow.setPixmap(QPixmap(filename))
        elif index == 2:
            filename, _ = QFileDialog.getOpenFileName(self, "选择视频", os.getcwd(), "Videos (*.avi *.mp4);;All (*)")
            self.capture = cv.VideoCapture(filename)
            self.fps = self.capture.get(cv.CAP_PROP_FPS)  # 获得视频帧率
            self.timer.timeout.connect(self.slot_video_display) # 设置ontime循环槽函数
            flag, self.frame = self.capture.read()  # 显示视频第一帧
            if flag:
                self.pic_show(self.label_PrePicShow, self.frame)

    def onbuttonclick_selectDetect(self, index):
        """选择二维检测方式"""
        if index == 1:
            self.worker = YoloDetectWork()
        elif index == 2:
            return
        elif index == 3:
            self.worker = MaskDetectWork()
        else:
            return
        self.worker.moveToThread(self.thread)
        self.worker.send_detect_img.connect(self.slot_pic_flush)  # 二维检测中的信号与槽函数self.pic_flush连接
        self.startThreadSignal.connect(self.worker.work)  # 把触发子线程信号和处理函数绑定

    def onbuttonclick_videodisplay(self):
        """显示视频控制函数, 用于连接定时器超时触发槽函数"""
        if self.pushButton_3Ddetect.text() == "检测":
            self.timer.start(1000 / self.fps)
            self.pushButton_3Ddetect.setText("暂停")
        else:
            self.timer.stop()
            self.pushButton_3Ddetect.setText("检测")

    def slot_video_display(self):
        """定时器超时触发槽函数, 在label上显示每帧视频, 防止卡顿"""
        if not self.worker.flush:
            return
        flag, self.frame = self.capture.read()
        if flag:
            self.worker.img = self.frame
            self.thread.start()
            self.startThreadSignal.emit()  # 开始线程 发送信号
        else:
            self.capture.release()
            self.timer.stop()

    def slot_pic_flush(self, img):
        """处理完之后的图像在对应label中刷新，对应刷新信号"""
        self.pic_show(self.label_PrePicShow, img)

    def pic_show(self, label, pic):
        """图片在对应label中显示"""
        qimage = QImage(pic.data, pic.shape[1], pic.shape[0], QImage.Format_RGB888).rgbSwapped()
        label.setPixmap(QPixmap.fromImage(qimage))

    def onbuttonclick_picdetect(self):
        pass

    def onbuttonclick_videodetect(self):
        pass

    def destroy_thread(self):
        """在关闭窗口后，关掉进程"""
        self.thread.quit()
        self.thread.wait()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = Main()
    win.show()
    print(app.exec_())
    sys.exit(app.exec_())


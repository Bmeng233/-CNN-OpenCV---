# -*-coding:utf-8-*-
import cv2
import os
import time
import random


class Gesture():

    def __init__(self, train_path, predict_path, gesture):
        self.blurValue = 5
        self.bgSubThreshold = 36
        self.train_path = train_path
        self.predict_path = predict_path
        self.threshold = 60
        self.gesture = gesture
        self.skinkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.x1 = 380
        self.y1 = 60
        self.x2 = 640
        self.y2 = 350

    def collect_gesture(self, ges, photo_num):
        photo_num = photo_num
        video = False
        count = 0
        # 读取默认摄像头
        cap = cv2.VideoCapture(1)
        # 设置捕捉模式
        cap.set(10, 200)
        # 背景减法及初始化
        BgModel = cv2.createBackgroundSubtractorMOG2(0, self.bgSubThreshold)

        while True:
            # 读取视频帧
            ret, frame = cap.read()
            # 镜像转换
            frame = cv2.flip(frame, 1)
            cv2.imshow("Video Stream", frame)
            # 双边滤波
            frame = cv2.bilateralFilter(frame, 5, 50, 100)
            # 定义roi区域，第一个为y的取值，第2个为x的取值
            frame = frame[self.y1:self.y2, self.x1:self.x2]
            cv2.imshow('ROI', frame)
            # 背景减法运动检测
            bg = BgModel.apply(frame, learningRate=0)
            # 图像边缘处理--腐蚀
            Fgmask = cv2.erode(bg, self.skinkernel, iterations=1)
            # 将原始图像与背景减法+腐蚀处理后的蒙版做"与"操作
            bitwise_and = cv2.bitwise_and(frame, frame, mask=Fgmask)
            # 灰度处理
            gray = cv2.cvtColor(bitwise_and, cv2.COLOR_BGR2GRAY)
            # 高斯滤波
            blur = cv2.GaussianBlur(gray, (self.blurValue, self.blurValue), 2)
            # 使用自适应阈值分割(adaptiveThreshold)
            thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            cv2.imshow('Reprocessing', thresh)
            Ges = cv2.resize(thresh, (100, 100))

            if video is True and count < photo_num:
                # 录制训练集
                cv2.imencode('.jpg', Ges)[1].tofile(self.train_path + '{}_{}.jpg'.format(str(random.randrange(1000, 100000)),str(ges)))
                count += 1
                print(count)
            elif count == photo_num:
                print('{}训练集手势录制成功，即将开始录制测试集，共{}张'.format(photo_num, int(photo_num*0.43)))
                time.sleep(3)
                count += 1
            elif video is True and photo_num < count < int(photo_num*1.43):
                cv2.imencode('.jpg', Ges)[1].tofile(self.predict_path + '{}_{}.jpg'.format(str(random.randrange(1000, 100000)),str(ges)))
                count += 1
                print(count)
            elif video is True and count >= int(photo_num*1.43):
                video = False
                ges += 1
                if ges < len(self.gesture):
                    print('本手势录制完成，按空格键继续')
                else:
                    print('完成！')

            k = cv2.waitKey(10)
            if k == 27:
                break

            elif k == ord(' '):  # 录制手势
                video = True
                count = 0

            elif k == ord('b'):
                BgModel = cv2.createBackgroundSubtractorMOG2(0, self.bgSubThreshold)
                print('背景重置成功')


if __name__ == '__main__':

    Gesturetype = ['FIST', 'LIKE', 'PEACE', 'STOP', 'ROCK']
    train_path = 'Gesture_train/'
    predict_path = 'Gesture_predict/'
    for path in [train_path, predict_path]:
        if not os.path.exists(path):
            os.mkdir(path)
    print(f'训练手势为：{Gesturetype}')

    # 初始化手势识别类
    Ges = Gesture(train_path, predict_path, Gesturetype)
    # 单个手势要录制的数量
    num = 500
    # 训练手势类别计数器
    x = 0
    # 调用启动函数
    Ges.collect_gesture(ges=x, photo_num=num)

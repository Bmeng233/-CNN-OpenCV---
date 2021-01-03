import cv2


class SaveGesture:
    def __init__(self):
        self.blurValue = 5
        self.bgSubThreshold = 36
        self.threshold = 60
        self.skinkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.x1 = 380
        self.y1 = 60
        self.x2 = 640
        self.y2 = 350

    def saveGesture(self):
        cameraCapture = cv2.VideoCapture(1)
        success, frame = cameraCapture.read()
        # 初始化
        bgModel = cv2.createBackgroundSubtractorMOG2(0, self.bgSubThreshold)
        if success is True:
            # 镜像转换
            frame = cv2.flip(frame, 1)
            # 双边滤波
            frame = cv2.bilateralFilter(frame, 5, 50, 100)
            frame = frame[self.y1:self.y2, self.x1:self.x2]
            # 运动检测
            bg = bgModel.apply(frame, learningRate=0)
            # 边缘处理
            fgmask = cv2.erode(bg, self.skinkernel, iterations=1)
            # 背景减法
            bitwise_and = cv2.bitwise_and(frame, frame, mask=fgmask)
            # 灰度处理
            gray = cv2.cvtColor(bitwise_and, cv2.COLOR_BGR2GRAY)
            # 高斯滤波
            blur = cv2.GaussianBlur(gray, (self.blurValue, self.blurValue), 2)
            # 自适应阈值分割
            thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            cv2.imshow('Video Stream', thresh)
            Ges = cv2.resize(thresh, (640, 480))
            cv2.imwrite("E:/Python project/machine-learning/data/New_test_image/" + "test.jpg", Ges)
        testImg = cv2.imread('E:/Python project/machine-learning/data/New_test_image/test.jpg')
        img_roi_y = 30
        img_roi_x = 200
        img_roi_height = 450
        img_roi_width = 450
        img_roi = testImg[img_roi_y:(img_roi_y + img_roi_height), img_roi_x:(img_roi_x + img_roi_width)]
        cv2.imwrite("E:/Python project/machine-learning/data/New_test_image/roi/" + "img_roi.jpg", img_roi)
        cv2.waitKey(0)
        cv2.destroyWindow("[ROI_Img]")

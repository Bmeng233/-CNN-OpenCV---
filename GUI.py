import sys
import time
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import Qt, pyqtSignal, QDateTime, QThread
from PyQt5.QtGui import QIcon
from PyQt5.QtGui import QPixmap, QPalette
from Realtime_Processing import *
from Realtime_Test import *
from GUI_setting import *


class MyMainWindow(QMainWindow,Ui_MainWindow):
    def __init__(self):
        super(MyMainWindow, self).__init__()
        self.setupUi(self)
        self.initUI()

    def initUI(self):
        self.GetGestureButton.clicked.connect(self.GetGesture)
        self.JudgeButton.clicked.connect(self.JudgeGesture)

        # 窗口设置
        self.setWindowTitle('基于CNN和OpenCV的手势识别系统-HFUT机器学习大作业')
        self.setWindowIcon(QIcon('./GUIico/FRAME.ico'))
        self.resize(750, 485)
        self.initxianceng()
        self.CloseButton.setProperty('color', 'gray')
        self.GetGestureButton.setProperty('color', 'same')
        self.JudgeButton.setProperty('color', 'same')

    def GetGesture(self):
        self.LitResultlabel.setText("")
        self.ImaResultlabel.setPixmap(QPixmap('./GUIico/BACKGROUND.ico'))
        self.LitResultlabel.setAutoFillBackground(False)
        sg = SaveGesture()
        sg.saveGesture()
        self.LitResultlabel.setText("已将图像保存电脑本地")
        self.LitResultlabel.setAlignment(Qt.AlignCenter)

    def JudgeGesture(self):
        global gesture_action
        self.LitResultlabel.setText("正在通过CNN识别图像...")
        self.LitResultlabel.setAlignment(Qt.AlignCenter)
        QApplication.processEvents()
        gesture_num = evaluate_one_image()
        if gesture_num == 1:
            gesture_action = "1"
            self.result_show_1()
        elif gesture_num == 2:
            gesture_action = "2"
            self.result_show_2()
        elif gesture_num == 3:
            gesture_action = "3"
            self.result_show_3()
        elif gesture_num == 4:
            gesture_action = "4"
            self.result_show_4()
        elif gesture_num == 5:
            gesture_action = "5"
            self.result_show_5()



    def result_show_1(self):
        self.LitResultlabel.setText("该手势为FIST")
        self.LitResultlabel.setAutoFillBackground(True)
        palette = QPalette()
        palette.setColor(QPalette.Window, Qt.lightGray)
        self.LitResultlabel.setPalette(palette)
        self.ImaResultlabel.setPixmap(QPixmap('./GUIico/1FIST.ico'))
        self.LitResultlabel.setAlignment(Qt.AlignCenter)
        self.ImaResultlabel.setAlignment(Qt.AlignCenter)

    def result_show_2(self):
        self.LitResultlabel.setText("该手势为LIKE")
        self.LitResultlabel.setAutoFillBackground(True)
        palette = QPalette()
        palette.setColor(QPalette.Window, Qt.lightGray)
        self.LitResultlabel.setPalette(palette)
        self.ImaResultlabel.setPixmap(QPixmap('./GUIico/2LIKE.ico'))
        self.LitResultlabel.setAlignment(Qt.AlignCenter)
        self.ImaResultlabel.setAlignment(Qt.AlignCenter)

    def result_show_3(self):
        self.LitResultlabel.setText("该手势为PEACE")
        self.LitResultlabel.setAutoFillBackground(True)
        palette = QPalette()
        palette.setColor(QPalette.Window, Qt.lightGray)
        self.LitResultlabel.setPalette(palette)

        self.ImaResultlabel.setPixmap(QPixmap('./GUIico/3PEACE.ico'))
        self.LitResultlabel.setAlignment(Qt.AlignCenter)
        self.ImaResultlabel.setAlignment(Qt.AlignCenter)

    def result_show_4(self):
        self.LitResultlabel.setText("该手势为STOP")
        self.LitResultlabel.setAutoFillBackground(True)
        palette = QPalette()
        palette.setColor(QPalette.Window, Qt.lightGray)
        self.LitResultlabel.setPalette(palette)

        self.ImaResultlabel.setPixmap(QPixmap('./GUIico/4STOP.ico'))
        self.LitResultlabel.setAlignment(Qt.AlignCenter)
        self.ImaResultlabel.setAlignment(Qt.AlignCenter)

    def result_show_5(self):
        self.LitResultlabel.setText("该手势为ROCK")
        self.LitResultlabel.setAutoFillBackground(True)
        palette = QPalette()
        palette.setColor(QPalette.Window, Qt.lightGray)
        self.LitResultlabel.setPalette(palette)

        self.ImaResultlabel.setPixmap(QPixmap('./GUIico/5ROCK.ico'))
        self.LitResultlabel.setAlignment(Qt.AlignCenter)
        self.ImaResultlabel.setAlignment(Qt.AlignCenter)

    def initxianceng(self):
        self.backend = BackendThread()
        self.backend.update_date.connect(self.handleDisplay)
        self.backend.start()


    def handleDisplay(self, data):
        self.statusBar().showMessage(data)


class BackendThread(QThread):
    update_date = pyqtSignal(str)

    def run(self):
        while True:
            date = QDateTime.currentDateTime()
            currTime = date.toString('yyyy-MM-dd hh:mm:ss')
            self.update_date.emit(str(currTime))
            time.sleep(1)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWin = MyMainWindow()
    myWin.setObjectName('Window')

    myWin.show()
    sys.exit(app.exec_())


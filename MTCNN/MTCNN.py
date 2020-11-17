#Anthur:龙文汉
#Data:2020.11.17
#discrpnation:特征点提取图片

#加载模型
import sys
import cv2
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import threading
from threading import Thread,Lock

sys.path.append('./')#系统添加当前路径
from Detection.MtcnnDetector import MtcnnDetector
from Detection.detector import Detector
from Detection.fcn_detector import FcnDetector
from mtcnn_model import P_Net, R_Net, O_Net
from loader import TestLoader


#检测模型类
class Detect:
    def __init__(self):
        self.test_mode = "ONet"
        self.thresh =[0.9,0.6,0.7]#阈值
        self.min_face_size = 24
        self.stride = 2
        self.slide_window = False
        self.shuffle = False
        self.detectors = [None,None,None]
        self.prefix = ['./PNet_landmark/PNet', './RNet_landmark/RNet', './ONet_landmark/ONet']

        self.epoch = [18,14,16]
        self.batch_size = [2048,256,16]
        self.model_path = ['%s-%s'%(x,y) for x,y in zip(self.prefix,self.epoch)]

    def Load_pnet_model(self):
        if self.slide_window:
            PNet = Detector(P_Net,12,self.batch_size[0],self.model_path[0])
        else:
            PNet = FcnDetector(P_Net,self.model_path[0])
            print(self.model_path[0])
        self.detectors[0] = PNet

    def Load_rnet_model(self):
        if self.test_mode in ['RNet","ONet']:
            RNet = Detector(R_Net,24,self.batch_size[1],self.model_path[1])
            self.detectors[1] = RNet

    def Load_onet_model(self):
        if self.test_mode == "ONet":
            ONet = Detector(O_Net,48,self.batch_size[2],self.model_path[2])
            self.detectors[2] = ONet

    def Load(self):
        self.Load_pnet_model()
        self.Load_rnet_model()
        self.Load_onet_model()
        return MtcnnDetector(detectors=self.detectors,min_face_size=self.min_face_size,
                             stride=self.stride,threshold=self.thresh,slide_window=self.slide_window)
#开启线程摄像头
class OpenCV:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.Frame = []
        self.is_open = True
        self.status = 0

    def Start(self):
        print("thread started with videocapture" )
        threading.Thread(target=self.QueryFrame,daemon=True,args=()).start()#加括号的问题

    def Stop(self):
        self.is_open = False
        print("threashed stop!")

    def GetFrame(self):
        return self.Frame

    def QueryFrame(self):
        while(self.is_open):
            self.status,self.Frame = self.cap.read()
        self.cap.release()

#实现类，通过摄像头和模型进行检测
class Begin_detect:
    def __init__(self):
        self.capture = OpenCV()
        self.dete = Detect()
        self.detector = self.dete.Load()
        self.lock = Lock()

    def Start(self):
        print("Detect thread begin")
        threading.Thread(target=self.Show_viedo,daemon=True,args=()).start()

    def Detect_five_point(self,frame):
        # test_data = TestLoader(frame)
        all_boxes, landmarks = self.detector.single_detect_face(frame)
        count = 0
        image = frame
        for bbox, landmark in zip(all_boxes[count], landmarks[count]):
            cv2.putText(image, str(np.round(bbox[4], 2)), (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_TRIPLEX, 1,
                            color=(255, 0, 255))
            cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 7)

        for landmark in landmarks[count]:
            for i in range(int(len(landmark) / 2)):
                    cv2.circle(image, (int(landmark[2 * i]), int(int(landmark[2 * i + 1]))), 3, (0, 0, 255))
        return image

    def Show_viedo(self):
        self.capture.Start()
        time.sleep(1)
        while(True):
            self.frame = self.capture.GetFrame()

            # 检测分割
            self.lock.acquire()
            try:
                self.after_frame = self.Detect_five_point(self.frame)
            except:
                self.after_frame = self.frame
            self.lock.release()

            #显示
            cv2.namedWindow("video_before(with 'q' exit)" , 0)  # 调节窗口
            cv2.imshow("video_before(with 'q' exit)", self.frame)

            cv2.namedWindow("video_after(with 'q' exit)", 0)  # 调节窗口
            cv2.imshow("video_after(with 'q' exit)", self.frame)

            flag = cv2.waitKey(1)  # 1毫秒
            # 输入 q 键退出
            if flag == ord('q'):
                cv2.destroyWindow("video_before(with 'q' exit)")
                cv2.destroyWindow("video_before(with 'q' exit)")

                cv2.destroyWindow("video_after(with 'q' exit)")
                cv2.destroyWindow("video_after(with 'q' exit)")
                break
        self.capture.Stop()

if __name__ == "__main__":
    begin = Begin_detect()
    begin.Start()
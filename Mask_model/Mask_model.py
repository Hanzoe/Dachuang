#双线程

import sys
import cv2
import paddlehub as hub
import time
import threading
from threading import Thread,Lock
import os

lock = Lock()

def Begin_Detect():   #调用检测函数
    module = hub.Module(name="pyramidbox_lite_mobile_mask", version='1.1.0')  #更改模型
    # way = "http://admin:admin@192.168.1.5:8081/"  # 此处@后的ipv4 地址需要改为app提供的地址
    way = "0" #获取打开方式
    Detect(module,way)      #开始检测

class opencv_capture:  #创建多线程，保证视屏检测流畅
    def __init__(self,module,URL):
        self.Frame =[]      #返回帧
        self.status = False     #状态码
        self.isstop = False     #停止码
        self.url = str(URL)
        # self.module = hub.Module(name="pyramidbox_lite_mobile_mask", version='1.1.0')  #更改模型
        self.module = module
        self.capture = cv2.VideoCapture(URL)  # 调用视屏流

    def start(self):        #开始线程
        print("thread started" + " with "+str(self.url))
        threading.Thread(target=self.queryframe,daemon=True,args=()).start()

    def stop(self):     #停止线程
        self.isstop = True
        self.capture.release()
        print("thread stoped" + " with "+str(self.url))

    def getframe(self):     #返回帧
        return self.Frame


    def queryframe(self):       #从视频流获取每一帧
        while(not self.isstop):
            self.status,self.Frame = self.capture.read()
        self.capture.release()  #释放

class detect_show:
    def __init__(self,module,way):
        self.module = module
        self.cap = opencv_capture(module,way)
        self.way = way
        self.rect = []

    def start(self):
        print("while trhread")
        threading.Thread(target=self.out_show, daemon=True, args=()).start()

    def frame_detect(self):  # 检测帧
        input_dict = {"data": [self.frame]}
        results = self.module.face_detection(data=input_dict)  # 调用模型
        if len(results) != 0:  # 多张人脸情况
            label = results[0]['data']['label']
            x1 = int(results[0]['data']['left'])
            y1 = int(results[0]['data']['top'])
            x2 = int(results[0]['data']['right'])
            y2 = int(results[0]['data']['bottom'])
            self.rect = (x1, y1, x2, y2)

            for result in results:
                label = result['data']['label']
                confidence = result['data']['confidence']

                top, right, bottom, left = int(result['data']['top']), int(result['data']['right']), int(
                    result['data']['bottom']), int(result['data']['left'])

                color = (0, 255, 0)
                if label == 'NO MASK':
                    color = (0, 0, 255)

                # 进行绘图标记
                cv2.rectangle(self.frame, (left, top), (right, bottom), color, 3)
                cv2.putText(self.frame, label + ":" + str(confidence), (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            color, 2)

    def out_show(self):
        self.cap.start()
        time.sleep(1)
        output_dir = "./photo"
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        while(1):
            self.frame = self.cap.getframe()
            lock.acquire()
            self.frame_detect()
            lock.release()
            # 显示
            frame_after = frame_cut(self.frame,self.rect)
            cv2.namedWindow("video(with 'q' exit) with" + str(self.way), 0)  # 调节窗口
            cv2.imshow("video(with 'q' exit) with" + str(self.way), self.frame)

            cv2.namedWindow("after video(with 'q' exit,with 's' save frame) with" + str(self.way), 0)  # 调节窗口
            cv2.imshow("after video(with 'q' exit,with 's' save frame) with" + str(self.way), frame_after)

            flag = cv2.waitKey(1)  # 1毫秒
            if flag == ord('s'):
                name = input("please input your name:")
                output_path = os.path.join(output_dir, "%s .jpg" % name)
                print(output_path)
                if cv2.imwrite(output_path, frame_after) == True:
                    print("accessed")
                else:
                    print("default")
                break
            # 输入 q 键退出
            if flag == ord('q'):
                cv2.destroyWindow("video(with 'q' exit) with" + str(self.way))
                cv2.destroyWindow("after video(with 'q' exit,with 's' save frame) with" + str(self.way))
                break

        self.cap.stop()

# def detect_mask(frame, module):
#     input_dict = {"data": [frame]}
#     results = module.face_detection(data=input_dict)
#     if results != []:
#         label = results[0]['data']['label']
#         x1 = int(results[0]['data']['left'])
#         y1 = int(results[0]['data']['top'])
#         x2 = int(results[0]['data']['right'])
#         y2 = int(results[0]['data']['bottom'])
#         rect = (x1, y1, x2, y2)
#         return rect, label
#     else:
#         return (0, 0, 0, 0), None

def frame_cut(frame,rect):
    # if results != []:
    #     x1 = int(results[0]['data'][0][1][0])
    #     y1 = int(results[0]['data'][0][19][1])
    #     x2 = int(results[0]['data'][0][15][0])
    #     y2 = int(results[0]['data'][0][15][1])
    #     h = y2 - y1
    #     y1 = int(y1 - (h / 2))
    #     cut_img = frame[y1:y2, x1:x2]
    #     if x1 > 0 or y1 > 0:
    #         return cut_img
    #     else:
    #         return frame
    # else:
    #     return frame
    # 按比列切割
    x1 = rect[0]
    y1 = rect[1]
    x2 = rect[2]
    y2 = rect[3]
    h = y2 - y1
    w = x2 - x1
    y2_0 = y2 - int((y2 - y1) / 1.8)
    cut_img = frame[y1:y2_0, x1:x2]
    if x1 <= 0 or y1 <= 0:
        return None

    return cv2.cvtColor(cut_img, cv2.COLOR_BGR2GRAY)

def frame_detect(module,frame):  # 检测帧
    input_dict = {"data": [frame]}
    results = module.face_detection(data=input_dict)  # 调用模型
    if len(results) != 0:  # 多张人脸情况
        for result in results:
            label = result['data']['label']
            confidence = result['data']['confidence']

            top, right, bottom, left = int(result['data']['top']), int(result['data']['right']), int(
                result['data']['bottom']), int(result['data']['left'])

            color = (0, 255, 0)
            if label == 'NO MASK':
                color = (0, 0, 255)

            # 进行绘图标记
            cv2.rectangle(frame, (left, top), (right, bottom), color, 3)
            cv2.putText(frame, label + ":" + str(confidence), (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        color, 2)
    return frame

def detect_number(module,way):
    ds1 = detect_show(module,way)
    # ds2 = detect_show(module,0)
    # ds3 = detect_show(module,1)

    # ds2.start()
    # ds3.start()
    ds1.start()

def Detect(module,way):
    #判断打开文件类型
    if way[-1] == 'g':
        img = cv2.imread(way)
        test_img_path = []
        test_img_path.append(way)
        input_dict = {"image": test_img_path}
        results = module.face_detection(data=input_dict)
        if len(results) != 0:
            for result in results:
                label = result['data']['label']
                confidence = result['data']['confidence']

                top, right, bottom, left = int(result['data']['top']), int(result['data']['right']), int(
                    result['data']['bottom']), int(result['data']['left'])

                color = (0, 255, 0)
                if label == 'NO MASK':
                    color = (0, 0, 255)

                cv2.rectangle(img, (left, top), (right, bottom), color, 3)
                cv2.putText(img, label + ":" + str(confidence), (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            color, 2)
        # 预测结果展示
        cv2.namedWindow("photo", 0)  # 调节窗口
        cv2.imshow("photo",img)
        cv2.wait
    else:
        if way.isnumeric():
            way = int(way)
            if way<1:
                way = 0
        detect_number(module,way)

if __name__ == '__main__':
    Begin_Detect()


import datetime
import enum
import threading
import time

import cv2
# import matplotlib.pyplot as plt
import numpy as np
import requests

from camera_thread import CameraThread
from mobilenetv4vector import MobilenetV4Vector
from ocr_sender import OCRSender
from vector_recoder import VectorRecorder
from yolodetect import YoloDetect

class PacketForServer:
    def __init__(self):
        self.ip_address = '172.23.161.182'
        self.port = '9000'
        self.url_path = '/sendphotocheck/api/register_car_photo/'
        self.apikey = "OrH04mwu.PbbcKY9n1OFye1irZW7NuzbJVPEueyHo"

        self.base_url = 'http://' + self.ip_address + ':' + self.port + self.url_path
        self.headers={'Accept':'application/json', 'X-Api-Key': self.apikey}
    
    def send(self, image, bbox):
        ret, encoded_img = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        if not ret:
            print("fail encode numpy shape to jpg shape")
            return False

        url = self.base_url
        if url[-1] != "/":
            url += "/"

        send_files = {
            'car_image': (f'{datetime.datetime.now()}_cam.jpg', encoded_img.tobytes(), 'image/jpeg'),
        }

        send_data = {
            'car_image_width': image.shape[1],
            'car_image_height': image.shape[0],
            'bbox_left_x': bbox[0],
            'bbox_right_x': bbox[2],
            'bbox_above_y': bbox[1],
            'bbox_below_y': bbox[3],
        }
        
        try:
            requests.post(url, data=send_data, files=send_files, headers = self.headers)
        except requests.RequestException as e:
            print(e)
            return False

        return True

class CarDetectionStatusEnum(enum.Enum):
    DETECT = enum.auto()
    DETECTING = enum.auto()
    NOT_DETECT = enum.auto()
    NOT_DETECTING = enum.auto()

class CarDetectionStatus:
    def __init__(self, init_status = CarDetectionStatusEnum.NOT_DETECT):
        self.status = init_status
        self.count = 0

        self.pre_status = init_status
        self.pre_count = 0

        self.max_count = 5

    def update(self, is_detect):
        self.pre_status = self.status
        self.pre_count = self.count
        if is_detect:
            if self.status == CarDetectionStatusEnum.DETECTING:
                self.count += 1
            elif self.status == CarDetectionStatusEnum.NOT_DETECT:
                self.status = CarDetectionStatusEnum.DETECTING
                self.count = 0
            elif self.status == CarDetectionStatusEnum.NOT_DETECTING:
                if self.pre_status == CarDetectionStatusEnum.DETECT:
                    self.status = CarDetectionStatusEnum.DETECT
                else:
                    self.status = CarDetectionStatusEnum.DETECTING
                    self.count = 0
            
            if self.count >= self.max_count:
                self.status = CarDetectionStatusEnum.DETECT
                self.count = 0
        else:
            if self.status == CarDetectionStatusEnum.NOT_DETECTING:
                self.count += 1
            elif self.status == CarDetectionStatusEnum.DETECT:
                self.status = CarDetectionStatusEnum.NOT_DETECTING
                self.count = 0
            elif self.status == CarDetectionStatusEnum.DETECTING:
                if self.pre_status == CarDetectionStatusEnum.NOT_DETECT:
                    self.status = CarDetectionStatusEnum.NOT_DETECT
                else:
                    self.status = CarDetectionStatusEnum.NOT_DETECTING
                    self.count = 0
            
            if self.count >= self.max_count:
                self.status = CarDetectionStatusEnum.NOT_DETECT
                self.count = 0

    def get(self):
        return self.status, self.status == self.pre_status

class SenderThread:
    def __init__(self):
        self.yolo = YoloDetect(yolo_path = "./weights/last.onnx")
        self.mobilenetv4 = MobilenetV4Vector()
        self.camera_thread = CameraThread()
        # self.packet_for_server = PacketForServer()
        self.vector_recorder = VectorRecorder()
        self.ocr_sender = OCRSender()

        self.car_detection_status = CarDetectionStatus(init_status = CarDetectionStatusEnum.DETECT if self.vector_recorder.is_saved_vector() else CarDetectionStatusEnum.NOT_DETECT)
        self.thread = None
        self.thread_run = False
    
    def start(self):
        self.camera_thread.start()
        self.thread = threading.Thread(target=self.loop)
        self.thread.start()
    
    def stop(self):
        self.thread_run = False
        if self.thread:
            self.thread.join()
        self.camera_thread.stop()

    def loop(self):
        self.thread_run = True

        while self.thread_run:
            img = self.camera_thread.next()

            if img is None:
                print("fail getting frame.")
                time.sleep(1)
                continue
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            bbox, score = self.yolo.get_xyxy(img.copy())
            self.car_detection_status.update(score is not None)

            detection_status, status_is_updated = self.car_detection_status.get()

            if detection_status == CarDetectionStatusEnum.DETECT:
                img_vector = self.mobilenetv4.get_vector(img.copy(), bbox)

                if self.vector_recorder.is_same_vector(img_vector):
                    continue

                # ret = self.packet_for_server.send(img, bbox)
                ocr_result = self.ocr_sender.send(img.copy(), bbox)
                if ocr_result is not None:
                    pass
                    # TODO add directus code here
                self.vector_recorder.save(img_vector)

                # if not ret:
                #     self.stop()
            elif detection_status == CarDetectionStatusEnum.NOT_DETECT and status_is_updated:
                # TODO add directus code here
                self.vector_recorder.reset()

        self.thread_run = False
    
    def is_running(self):
        return self.thread_run

if __name__=="__main__":
    sender_thread = SenderThread()
    sender_thread.start()

    while True:
        if sender_thread.is_running():
            time.sleep(1)
        else:
            break
        




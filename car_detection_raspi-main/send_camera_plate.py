import numpy as np
import cv2
import time

from yolodetect import YoloDetect
from mobilenetv4vector import MobilenetV4Vector
from camera_thread import CameraThread3

# import matplotlib.pyplot as plt

import requests
import datetime

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
        
        requests.post(url, data=send_data, files=send_files, headers = self.headers)

        return True

class VectorRecorder:
    def __init__(self):
        self.recorded_vector = None
    
    def save(self, vector):
        self.recorded_vector = vector

    def calc_distance(self, vector):
        if self.recorded_vector is None:
            return float('inf')
        
        return ((vector - self.recorded_vector) ** 2).mean()
    
    def is_same_vector(self, vector, boundary = 10.0):
        return self.calc_distance(vector) < boundary

if __name__=="__main__":
    yolo = YoloDetect(yolo_path = "./weights/last.onnx")
    mobilenetv4 = MobilenetV4Vector()
    camera_thread = CameraThread3()
    camera_thread.start()

    packet_for_server = PacketForServer()
    vector_recorder = VectorRecorder()

    detect = False
    number_of_continuous_detect = 0

    while True:
        img = camera_thread.next()

        if img is None:
            print("fail getting frame.")
            time.sleep(1)
            continue
        # else:
        #     plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        #     plt.axis('off')
        #     plt.show()
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        bbox, score = yolo.get_xyxy(img.copy())
        if score is None:
            detect = False
            number_of_continuous_detect = 0
        else:
            detect = True
            number_of_continuous_detect += 1

        if number_of_continuous_detect <= 5:
            continue

        img_vector = mobilenetv4.get_vector(img.copy(), bbox)

        if vector_recorder.is_same_vector(img_vector):
            continue

        packet_for_server.send(img, bbox)
        vector_recorder.save(img_vector)


        key = cv2.waitKey(1)
        if key != -1:
            break
    
    camera_thread.stop()





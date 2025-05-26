import os
import time

import cv2
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import ComputerVisionOcrErrorException, OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials

class OCRSender:
    def __init__(self):
        self.key = os.environ["VISION_KEY"]
        self.endpoint = os.environ["VISION_ENDPOINT"]

    def send(self, image, bbox):
        client = ComputerVisionClient(self.endpoint, CognitiveServicesCredentials(self.key))

        ret, encoded_img = cv2.imencode(".jpg", cv2.cvtColor(image[bbox[1]:bbox[3], bbox[0]:bbox[2]], cv2.COLOR_RGB2BGR))
        if not ret:
            print("failed encode image.")
            return None
        
        try:
            recognize_results = client.read_in_stream(encoded_img.tobytes(), language = "ja", raw = True)
        except ComputerVisionOcrErrorException as e:
            print("errors : ", e.response)
            return None
        
        operation_location_remote = recognize_results.headers["Operation-Location"]
        operation_id = operation_location_remote.split("/")[-1]

        while True:
            get_text_result = client.get_read_result(operation_id)
            if get_text_result.status not in ["notStarted", "running"]:
                break
            time.sleep(1)

        if get_text_result.status == OperationStatusCodes.succeeded:
            res = []
            for text_result in get_text_result.analyze_result.read_results:
                for line in text_result.lines:
                    res.append(line.text)
            return res
        else:
            print("failed OCR.")
            return None

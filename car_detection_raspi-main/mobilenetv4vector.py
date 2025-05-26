import cv2
import numpy as np
import onnxruntime

class MobilenetV4Vector:
    def __init__(self, mobilenetv4_path = "./weights/latest_k_0.onnx"):
        self.mobilenetv4_sess = onnxruntime.InferenceSession(mobilenetv4_path)
        self.mobilenetv4_width = 224
        self.mobilenetv4_height = 224
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
    
    def get_vector(self, image, bbox):
        # preprocess begin
        input_image = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        input_image = cv2.resize(input_image, dsize=(self.mobilenetv4_width, self.mobilenetv4_height))
        input_image = (input_image / 255 - self.mean) / self.std
        input_image = input_image.transpose(2, 0, 1)
        input_image = np.expand_dims(input_image, axis=0).astype("float32")
        # preprocess end

        mobilenetv4_input_name = self.mobilenetv4_sess.get_inputs()[0].name
        mobilenetv4_output_name = self.mobilenetv4_sess.get_outputs()[0].name
        mobilenetv4_result = self.mobilenetv4_sess.run([mobilenetv4_output_name], {mobilenetv4_input_name : input_image})[0]

        # postprocess begin
        mobilenetv4_result = mobilenetv4_result[0]
        return mobilenetv4_result
    
    def is_same_plate(self, vec0, vec1):
        return (vec0 - vec1) ** 2 < 10.0
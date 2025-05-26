import cv2
import numpy as np
import onnxruntime

class YoloDetect:
    def __init__(self, yolo_path = "./weights/last.onnx"):
        self.yolo_sess = onnxruntime.InferenceSession(yolo_path)
        self.yolo_width = self.yolo_sess.get_inputs()[0].shape[2]  # yolov11であれば640
        self.yolo_height = self.yolo_sess.get_inputs()[0].shape[3] # yolov11であれば640
    
    def get_xyxy(self, image):
        # モデルの指定したサイズにリサイズする
        image_height, image_width = image.shape[:2]
        image = cv2.resize(image, dsize=(self.yolo_width, self.yolo_height))

        # モデルの入力したサイズから元画像のサイズの倍率計算
        x_factor = image_width / self.yolo_width
        y_factor = image_height / self.yolo_height

        # DEBUG
        # import matplotlib.pyplot as plt
        # plt.imshow(image)
        # plt.axis('off')
        # plt.show()

        # 正規化や軸の変換などモデルに対応した形式に変換する
        input_image = image / 255
        input_image = input_image.transpose(2, 0, 1)
        input_image = np.expand_dims(input_image, axis=0).astype("float32")

        # yoloで推論する
        yolo_input_name = self.yolo_sess.get_inputs()[0].name
        yolo_output_name = self.yolo_sess.get_outputs()[0].name
        yolo_result = self.yolo_sess.run([yolo_output_name], {yolo_input_name : input_image})[0]

        # yoloの結果を扱いやすいように変換する
        yolo_result = yolo_result[0]
        yolo_result = yolo_result.transpose(1, 0)
        yolo_result = np.ascontiguousarray(yolo_result)

        # yoloの結果をそれぞれ抜き出す
        x_center, y_center, width, height, score = yolo_result[:, 0], yolo_result[:, 1], yolo_result[:, 2], yolo_result[:, 3], yolo_result[:, 4]

        # 検出結果を扱いやすいように変換する
        x_min = (x_center - width / 2) * x_factor
        y_min = (y_center - height / 2) * y_factor
        width = width * x_factor
        height = height * y_factor

        # bbox形式に変換する
        bbox = np.stack((x_min, y_min, width, height), axis=1)

        # nms_indice = np.argmax(score)
        # bbox = bbox[nms_indice]
        # score = score[nms_indice]
        # print(bbox, score)
        # cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), (0, 255, 0), 2)
        # import matplotlib.pyplot as plt
        # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # plt.axis('off')
        # plt.show()

        # NMSにより必要なbboxのみ抽出する
        nms_indice = cv2.dnn.NMSBoxes(bbox.tolist(), score.ravel().tolist(), 0.25, 0.7, top_k=1)

        print(nms_indice)
        # クラス確率の高くないbboxしかなかった場合，Noneを返す
        if len(nms_indice) == 0:
            return None, None
        
        # クラス確率の最も高いbbox, score, xyxy座標を計算する
        bbox = bbox[nms_indice[0]]
        score = score[nms_indice[0]]
        x_min = int(bbox[0])
        y_min = int(bbox[1])
        x_max = int(bbox[0] + bbox[2])
        y_max = int(bbox[1] + bbox[3])
        # postprocess end

        print([x_min, y_min, x_max, y_max])

        # xyxy形式bboxとクラス確率を返す
        return [x_min, y_min, x_max, y_max], score
# car-detection
car_detection_raspi
ラズパイ上で，車のナンバープレートを検出し，サーバに画像を送るプログラムたち
![annotated_image](https://github.com/user-attachments/assets/01aeb924-6061-4b5f-a1e4-e223a44e9e71)

camera_thread.py : カメラから画像を取得し続ける．別スレッドで動作する．

directus_sender.py : directusと通信する．未実装．

movilenetv4vector.py : ナンバープレート画像の特徴ベクトルを保存する．サーバに余計な画像を送らないようにするために使用．weights/latest_k_0.onnxが重み．

ocr_sender.py : ocrを行うサーバと通信する．今のところazureと通信する．

send_camera_plate.py : raspi上で動作確認済み．旧実装．

send_camera_plate2.py : raspi上で動作未確認．新実装．

vector_recorder.py : ナンバープレート画像の特徴ベクトルを保存する．サーバに余計な画像を送らないようにするために使用．

yolodetect.py : 画像内からナンバープレート検出する．yolov11nをファインチューニング．weights/last.onnxが重み．
![image copy 3](https://github.com/user-attachments/assets/02e28b59-2689-47d6-9cf0-c4396440b2ca)
![image copy 2](https://github.com/user-attachments/assets/01435dd0-b3e6-4cf3-9485-d117ce0d8ef3)
![image copy](https://github.com/user-attachments/assets/a2883a5e-edda-412c-a2c0-ff945e342cff)
![cropped_image3](https://github.com/user-attachments/assets/73a05695-8de9-423a-b644-dd407b7cf747)
![cropped_image2](https://github.com/user-attachments/assets/c2739295-d05e-4b29-bbbc-5f7274574d88)
![cropped_image1](https://github.com/user-attachments/assets/a7c03e3e-9fb0-4007-80d9-30febc8c0ea5)

現在の問題点、AzereのOCRが今のところ不十分。暗い、質が悪い画像を読み取ると精度が悪くなってします。 他の方法を探す。 Mistral OCRが出てくる。これは既存のOCR、MMLLMより精度が高いらしい
![image](https://github.com/user-attachments/assets/49e91c81-14f9-4a7e-b9e9-8cfc8f3bee71)

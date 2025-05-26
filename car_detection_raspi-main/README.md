# car_detection_raspi

ラズパイ上で，車のナンバープレートを検出し，サーバに画像を送るプログラムたち

camera_thread.py : カメラから画像を取得し続ける．別スレッドで動作する．

directus_sender.py : directusと通信する．未実装．

movilenetv4vector.py : ナンバープレート画像の特徴ベクトルを保存する．サーバに余計な画像を送らないようにするために使用．weights/latest_k_0.onnxが重み．

ocr_sender.py : ocrを行うサーバと通信する．今のところazureと通信する．

send_camera_plate.py : raspi上で動作確認済み．旧実装．

send_camera_plate2.py : raspi上で動作未確認．新実装．

vector_recorder.py : ナンバープレート画像の特徴ベクトルを保存する．サーバに余計な画像を送らないようにするために使用．

yolodetect.py : 画像内からナンバープレート検出する．yolov11nをファインチューニング．weights/last.onnxが重み．

# 🚗 Car Detection on Raspberry Pi

このプロジェクトは，**Raspberry Pi上で車のナンバープレートを検出し，必要に応じて画像をサーバへ送信する**一連のシステムです．YOLOベースの物体検出と，特徴ベクトルを用いたフィルタリング処理，さらにOCRによる文字抽出を組み合わせた構成となっています．

---
## 🧭 背景

街中には多数の駐車場が存在していますが，自由に利用できる駐車場は限られています．たとえば，昼間のみ営業する店舗の駐車場は，夜間は空いているにもかかわらず，夜の混雑時には使用できないことが多く，これは大きな無駄と言えます．
また，従来のロック板やゲートバーなどを導入するには高額なコストがかかります．これらを安価に代替できる仕組みがあれば，未活用の土地の有効活用が進み，利便性の向上にもつながると考えました．

---

## 🎯 目的

- 小型コンピュータ（Raspberry Pi）上で動作するリアルタイムなナンバープレート検出システムを構築すること
- 特徴ベクトルに基づく画像のフィルタリングにより，不要な画像の送信を避け，ネットワークやサーバへの負荷を軽減すること
- これらの技術を活用することで，低コストかつ設置が容易な駐車場システムを実現し，一時的なイベントや時間帯を指定した土地活用を可能にすること

---

## 🚧 困難

- Raspberry Pi上で処理可能な軽量モデルでありながら，高い検出精度を維持する必要がある
- 低照度・ブレ・低解像度といった条件下ではOCRによる文字認識精度が大きく低下する
- 重複画像を送信してしまうと，サーバやストレージに不要な負荷がかかる
- 日本語対応かつ高精度なOCRは少なく，既存のサービスでは精度が十分でない

---

## 🛠️ 対処

- YOLOv11nをベースにした軽量物体検出モデルをファインチューニングし，エッジデバイスでも動作可能な推論処理を実現
- MobileNetV4を用いた特徴ベクトル抽出により，画像の類似度を判定し，重複画像の送信を防止
- OCRにはAzure OCRを利用しつつ，今後より高精度なMistral OCRへの切り替えを検討中、また、オンラインリソースを活用し、自分で学習を行うことも検討。

---


## 📚 目次

- [📁 ディレクトリ構成](#-ディレクトリ構成)
- [🔧 各スクリプトの説明](#-各スクリプトの説明)
- [🧠 使用技術とモデル](#-使用技術とモデル)
- [⚠️ 現在の課題](#️-現在の課題)
- [📄 ライセンス](#-ライセンス)

---

## 📁 ディレクトリ構成

```
car-detection/
├── car_detection_raspi/
│   ├── camera_thread.py
│   ├── directus_sender.py
│   ├── movilenetv4vector.py
│   ├── ocr_sender.py
│   ├── send_camera_plate.py
│   ├── send_camera_plate2.py
│   ├── vector_recorder.py
│   └── yolodetect.py
├── weights/
│   ├── latest_k_0.onnx
│   └── last.onnx
├── image copy*/              # 元画像
├── cropped_image*/           # 検出後の切り出し画像
└── README.md
```

---

## 🔧 各スクリプトの説明

| スクリプト名 | 説明 |
|--------------|------|
| `camera_thread.py` | カメラ映像をスレッドで継続取得するモジュール |
| `directus_sender.py` | Directusとの通信機能（未実装） |
| `movilenetv4vector.py` | ナンバープレート画像の特徴ベクトル抽出（類似画像除去） |
| `ocr_sender.py` | Azure OCR APIと通信し文字を読み取る |
| `send_camera_plate.py` | 旧バージョンの実装（Raspberry Pi上で動作確認済み） |
| `send_camera_plate2.py` | 新バージョン（未確認） |
| `vector_recorder.py` | ベクトル抽出と記録（`movilenetv4vector.py`と連携） |
| `yolodetect.py` | YOLOv11nベースでナンバープレート検出（`last.onnx`使用） |

---

## 🧠 使用技術とモデル

- **YOLOv11n** をファインチューニングしてナンバープレート検出
- 特徴抽出に **MobileNetV4** を使用し，重複画像の送信を抑制
- OCRは **Azure OCR API** を利用（今後Mistral OCR等への切替も検討中）

---

## ⚠️ 現在の課題

- Azure OCRが暗所や低品質画像に弱く，文字認識精度が不十分
- 高精度な代替OCRとして **Mistral OCR** などの導入を検討

---

## 📄 ライセンス

MITライセンス

---

 
 
 # car-detection

![annotated_image](https://github.com/user-attachments/assets/01aeb924-6061-4b5f-a1e4-e223a44e9e71)


元データ↓

![image copy 3](https://github.com/user-attachments/assets/02e28b59-2689-47d6-9cf0-c4396440b2ca)


![image copy 2](https://github.com/user-attachments/assets/01435dd0-b3e6-4cf3-9485-d117ce0d8ef3)


![image copy](https://github.com/user-attachments/assets/a2883a5e-edda-412c-a2c0-ff945e342cff)


検出後データ↓

![cropped_image3](https://github.com/user-attachments/assets/73a05695-8de9-423a-b644-dd407b7cf747)


![cropped_image2](https://github.com/user-attachments/assets/c2739295-d05e-4b29-bbbc-5f7274574d88)


![cropped_image1](https://github.com/user-attachments/assets/a7c03e3e-9fb0-4007-80d9-30febc8c0ea5)

![image](https://github.com/user-attachments/assets/49e91c81-14f9-4a7e-b9e9-8cfc8f3bee71)


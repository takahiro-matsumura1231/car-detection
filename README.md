# 🚗 Car Detection on Raspberry Pi

このプロジェクトは，**Raspberry Pi上で車のナンバープレートを検出し，必要に応じて画像をサーバへ送信する**一連のシステムです．YOLOベースの物体検出と，特徴ベクトルを用いたフィルタリング処理，さらにOCRによる文字抽出を組み合わせた構成となっています．

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


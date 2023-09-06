import tensorflow as tf
import numpy as np
import picamera
import picamera.array
import cv2

class YOLOModel:
    def __init__(self, model_path):
        # TensorFlow Liteモデルをロード
        self.interpreter = tf.lite.Interpreter(model_path="/home/twingella/autonomous/models/model.tflite")
        self.interpreter.allocate_tensors()

        # 入力と出力の詳細を取得
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # カメラの設定
        self.camera = picamera.PiCamera()
        self.camera.resolution = (320, 240) # 画像サイズの設定（モデルに合わせる）

    def preprocess_image(self, image):
        # 画像の前処理（サイズ変更、正規化など）
        resized_image = cv2.resize(image, (self.input_details[0]['shape'][2], self.input_details[0]['shape'][1]))
        normalized_image = resized_image / 255.0
        input_array = np.expand_dims(normalized_image, axis=0)
        return input_array.astype(self.input_details[0]['dtype'])

    def detect_objects(self):
        with picamera.array.PiRGBArray(self.camera) as stream:
            self.camera.capture(stream, format='bgr')
            image = stream.array

            # 画像の前処理
            input_array = self.preprocess_image(image)

            # 推論を実行
            self.interpreter.set_tensor(self.input_details[0]['index'], input_array)
            self.interpreter.invoke()

            # 結果を取得
            detection_boxes = self.interpreter.get_tensor(self.output_details[0]['index'])
            detection_classes = self.interpreter.get_tensor(self.output_details[1]['index'])
            detection_scores = self.interpreter.get_tensor(self.output_details[2]['index'])
            num_detections = int(self.interpreter.get_tensor(self.output_details[3]['index'])[0])

            return detection_boxes, detection_classes, detection_scores, num_detections

# 使用例
model_path = 'path/to/your/model.tflite'
yolo = YOLOModel(model_path)
boxes, classes, scores, num = yolo.detect_objects()
print('Detected objects:', num)

import picamera #カメラモジュール用

cap = picamera.PiCamera() #インスタンス生成
cap.resolution = (1920, 1080) #画像サイズの指定
cap.capture("capture.jpg") #撮影

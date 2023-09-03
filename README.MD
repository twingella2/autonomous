電子工作の練習

Raspberry Pi 4 Model Bをベースにして自動運転の電子工作カーを作る。

公式 Raspberry Pi camera module v3
Ultrasound sensor:
Line tracking sensor:
Servomotor x2: Camera + Ultrasound sensorの首振り

車体を現在は組み込みの通常のタイヤを４つつけて走らせる予定

自動運転による車体制御の自由度を高めるためにのちにオムニホイールのようなものに付け替える予定
360度サーボモーターと組み合わせて上下だけでなく平行移動・斜め移動をする予定。

-----------------------------------

必要あればモーター駆動部分をArduino Nano 33 BLEへと接続させてRaspberry Piと組み合わせて作る予定。

tensorflowliteに読み込ませるためSSD MobileNet V2を変換して利用

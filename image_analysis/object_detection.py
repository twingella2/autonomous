# import subprocess
# import numpy as np
# import tflite_runtime.interpreter as tflite
# from PIL import Image

# def main():
#     # Initialize the TFLite interpreter
#     interpreter = tflite.Interpreter(model_path='/home/twingella/autonomous/models/model.tflite')
#     interpreter.allocate_tensors()

#     # Get input and output details
#     input_details = interpreter.get_input_details()
#     output_details = interpreter.get_output_details()

#     # Get the expected shape of the input tensor
#     expected_shape = input_details[0]['shape']
#     print("Expected input shape:", expected_shape)

#     # Capture an image using libcamera
#     subprocess.run(["libcamera-still", "-o", "frame.jpg"])

#     # Load and preprocess the image
#     image = Image.open('frame.jpg').convert('RGB')
#     image_resized = image.resize((expected_shape[1], expected_shape[2]))
#     image_np = np.array(image_resized)

#     # Check if the captured image shape matches the expected input shape
#     print("Actual image shape:", image_np.shape)

#     if image_np.shape[:2] == tuple(expected_shape[1:3]):
#         input_array = np.expand_dims(image_np, axis=0)
#     else:
#         print(f"Cannot reshape image of shape {image_np.shape} to expected shape {expected_shape}")
#         exit(1)

#     # Perform inference
#     interpreter.set_tensor(input_details[0]['index'], input_array.astype(np.uint8))
#     interpreter.invoke()

#     # Get the output tensor
#     output_data = interpreter.get_tensor(output_details[0]['index'])

#     # Check the size of the output_data
#     output_size = len(output_data[0])

#     # Analyze the output (Example: get the index of the most confident object detected)
#     max_index = np.argmax(output_data[0])

#     if max_index < output_size:
#         print(f"Detected object: {output_data[0][max_index]}")
#     else:
#         print(f"Index out of bounds. Max index: {max_index}, Output size: {output_size}")

# if __name__ == "__main__":
#     main()

import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image

def main():
    # Initialize the TFLite interpreter
    interpreter = tflite.Interpreter(model_path='/home/twingella/autonomous/models/model.tflite')
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    expected_shape = input_details[0]['shape']
    print("Expected input shape:", expected_shape)

    # Load YOLO
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # Capture an image using OpenCV
    cap = cv2.VideoCapture(0)
    _, frame = cap.read()
    height, width, channels = frame.shape

    # Detecting objects using YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), font, 2, (255,255,255), 2)

    cv2.imshow("Image", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image

def main():
    cap = None
    try:
        # Initialize the TFLite interpreter
        interpreter = tflite.Interpreter(model_path='/home/twingella/autonomous/models/model.tflite')
        interpreter.allocate_tensors()

        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Initialize the video capture
        camera_id = 0  # Change this if your camera ID is different
        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            raise Exception("Error: Couldn't open the camera.")

        while True:
            ret, frame = cap.read()
            
            if not ret:
                break

            # Convert the frame to PIL Image and resize
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            expected_shape = input_details[0]['shape'][1:3]
            image_resized = image.resize(expected_shape)
            image_np = np.array(image_resized)

            # Prepare the input tensor
            input_array = np.expand_dims(image_np, axis=0)

            # Perform inference
            interpreter.set_tensor(input_details[0]['index'], input_array.astype(np.uint8))
            interpreter.invoke()

            # Get the output tensor
            output_data = interpreter.get_tensor(output_details[0]['index'])
            
            # Get the index of the most confident object detected
            max_index = np.argmax(output_data[0])
            
            # Replace with actual label if available
            label = f"Object detected: {max_index}"

            # Overlay the label on the frame
            cv2.putText(frame, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            # Display the frame
            cv2.imshow('Object Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        if cap:
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

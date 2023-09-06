import pygame.camera
import pygame.image
import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image

# Initialize the camera
pygame.camera.init()
cam = pygame.camera.Camera("/dev/video0", (640, 480))
cam.start()

# Initialize the TFLite interpreter
interpreter = tflite.Interpreter(model_path='/home/twingella/autonomous/models/model.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Capture a single image
image = cam.get_image()
pygame.image.save(image, 'test_image.jpg')

# Convert the image for TensorFlow Lite
image_np = np.array(pygame.surfarray.array3d(image))
image_pil = Image.fromarray(np.transpose(image_np, (1, 0, 2)).astype(np.uint8))
expected_shape = input_details[0]['shape'][1]

import numpy as np
import tensorflow.lite as tflite
import cv2
import os

# Load TFLite model
interpreter = tflite.Interpreter(model_path="D:/Miniproject/bolt/project/model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load a test image
image_path = "D:/Miniproject/bolt/project/test_image.jpg"
if not os.path.exists(image_path):
    print(f"Error: Image file not found at {image_path}")
    exit()

image = cv2.imread(image_path)
if image is None:
    print(f"Error: Could not read the image file at {image_path}")
    exit()

image = cv2.resize(image, (224, 224))
image = image.astype(np.float32) / 255.0  # Normalize

# Expand dimensions to match the model input shape
image = np.expand_dims(image, axis=0)

# Perform inference
interpreter.set_tensor(input_details[0]['index'], image)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])

# Get prediction
predicted_class = np.argmax(output_data)
labels = ["100 Rupees", "200 Rupees", "500 Rupees"]  # Adjust based on your dataset
print(f"Predicted Class: {labels[predicted_class]}")

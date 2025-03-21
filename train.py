import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

# Define dataset path
dataset_path = "D:/Miniproject/bolt/project/processed_dataset/"
image_size = (224, 224)
batch_size = 32
epochs = 10

# Data augmentation and preprocessing
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    validation_split=0.2  # 80% training, 20% validation
)

train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="training"
)

val_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation"
)

# Define CNN Model
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(224, 224, 3)),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Conv2D(64, (3, 3), activation="relu"),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Conv2D(128, (3, 3), activation="relu"),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(3, activation="softmax")  # 3 classes (100, 200, 500)
])

# Compile the model
model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# Train the model
model.fit(train_generator, validation_data=val_generator, epochs=epochs)

# Save the model
model.save("D:/Miniproject/bolt/project/model.h5")

# Convert model to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model
tflite_model_path = "D:/Miniproject/bolt/project/model.tflite"
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

print(f"Model training complete. TFLite model saved at: {tflite_model_path}")

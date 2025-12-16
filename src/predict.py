import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

IMG_SIZE = 128
MODEL_PATH = "Model/blood_group_cnn_model.h5"
TEST_IMAGE_PATH = "test_images/test1.jpg"
OUTPUT_PATH = "outputs/results.txt"

model = load_model(MODEL_PATH)

class_names = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']

img = cv2.imread(TEST_IMAGE_PATH)
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
img = img / 255.0
img = np.expand_dims(img, axis=0)

prediction = model.predict(img)
predicted_class = class_names[np.argmax(prediction)]

print("🩸 Predicted Blood Group:", predicted_class)

with open(OUTPUT_PATH, "w") as f:
    f.write(f"Predicted Blood Group: {predicted_class}\n")

print("✅ Result saved in outputs/results.txt")

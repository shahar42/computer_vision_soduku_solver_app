# Simple direct model test
import cv2
import numpy as np
from models.digit_recognizer import RobustDigitRecognizer

# Initialize recognizer
recognizer = RobustDigitRecognizer()
recognizer.load("data/models/digit_recognizer.h5")

# Test on a sample cell (assuming you have cell_images from your pipeline)
# Replace with an actual extracted cell
sample_cell = cell_images[0][0]  # From your pipeline

# Test CNN directly
processed = recognizer.cnn_recognizer._preprocess_cell(sample_cell)
is_empty = recognizer.cnn_recognizer._is_empty_cell(processed)

print(f"Empty detected: {is_empty}")
print(f"CNN confidence threshold: {recognizer.cnn_recognizer.confidence_threshold}")

if not is_empty:
    input_data = processed.reshape(1, 28, 28, 1)
    raw_pred = recognizer.cnn_recognizer.model.predict(input_data, verbose=0)[0]
    digit = np.argmax(raw_pred)
    confidence = np.max(raw_pred)
    
    print(f"Raw prediction: digit={digit}, confidence={confidence:.3f}")
    print(f"Passes threshold: {confidence >= recognizer.cnn_recognizer.confidence_threshold}")

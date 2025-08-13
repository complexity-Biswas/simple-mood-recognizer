import cv2
import numpy as np

def preprocess_image(face_image):
    """Preprocess face for model input"""
    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))
    normalized = resized / 255.0
    return np.reshape(normalized, (1, 48, 48, 1))

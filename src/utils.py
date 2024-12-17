import cv2
import numpy as np
from keras.models import load_model

def load_trained_model(model_path):
    return load_model(model_path)

def preprocess_face_image(face_image):
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    face_image = cv2.resize(face_image, (48, 48))
    face_image = face_image.astype('float32') / 255
    return face_image

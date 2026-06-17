# models/cnn_model.py
import os
import tensorflow as tf
from tensorflow.keras.models import load_model as keras_load_model
import numpy as np

def load_model(model_path):
    if os.path.exists(model_path):
        try:
            model = keras_load_model(model_path)
            print(f"Model loaded from {model_path}")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    else:
        print(f"No model found at {model_path}. Please train first.")
        return None

def predict_image(model, preprocessed_image):
    if model is None:
        raise ValueError("Model not loaded. Train the model first.")
    prediction = model.predict(preprocessed_image, verbose=0)
    confidence = float(prediction[0][0])
    if confidence >= 0.5:
        label = "Malignant"
        confidence = confidence
    else:
        label = "Benign"
        confidence = 1 - confidence
    return label, confidence
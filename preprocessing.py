import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import io

# Set this to match your trained model's expected input size
# If you trained with 224x224, use (224,224)
# If you later train a new model with 380x380, change to (380,380)
MODEL_INPUT_SIZE = (224, 224)

def preprocess_uploaded_image(filepath):
    """Load image, resize to MODEL_INPUT_SIZE, normalize, and add batch dimension."""
    img = image.load_img(filepath, target_size=MODEL_INPUT_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def preprocess_image_bytes(image_bytes):
    """Preprocess image from bytes."""
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize(MODEL_INPUT_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array
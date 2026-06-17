# train_local.py – now supports zip upload
import os
import sys
import zipfile
import tempfile
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import argparse

print("TensorFlow version:", tf.__version__)

# ========== CONFIGURATION ==========
IMG_SIZE = (224, 224)   # Change to (380,380) for high‑accuracy model
BATCH_SIZE = 32
EPOCHS = 5   # default, will be overridden by command line
MODEL_SAVE_PATH = 'models/saved/breast_cancer_model.h5'
# ===================================

def extract_zip(zip_path, extract_to):
    """Extract a zip file to a directory."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    # Find the folder that contains 'benign' and 'malignant' subdirectories
    for root, dirs, files in os.walk(extract_to):
        if 'benign' in dirs and 'malignant' in dirs:
            return root
    # If not found, use the extract_to path itself
    return extract_to

def prepare_data(dataset_path):
    images, labels = [], []
    if not os.path.exists(dataset_path):
        raise ValueError(f"Dataset path not found: {dataset_path}")
    print(f"Loading images from: {dataset_path}")
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                filepath = os.path.join(root, file)
                if 'benign' in root.lower():
                    label = 'benign'
                elif 'malignant' in root.lower():
                    label = 'malignant'
                else:
                    continue
                try:
                    img = load_img(filepath, target_size=IMG_SIZE)
                    img_array = img_to_array(img) / 255.0
                    images.append(img_array)
                    labels.append(label)
                except Exception as e:
                    print(f"Error loading {filepath}: {e}")
    print(f"Loaded {len(images)} images")
    if len(images) == 0:
        raise ValueError("No images found. Check that the folder contains 'benign' and 'malignant' subfolders.")
    benign_cnt = labels.count('benign')
    malignant_cnt = labels.count('malignant')
    print(f"Benign: {benign_cnt}, Malignant: {malignant_cnt}")
    le = LabelEncoder()
    labels_enc = le.fit_transform(labels)
    return np.array(images), np.array(labels_enc), le

def create_model():
    base_model = EfficientNetB4(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
    base_model.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    out = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=out)
    return model, base_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--zip', help='Path to a zip file containing benign and malignant folders')
    parser.add_argument('--path', help='Direct path to dataset folder (optional)')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    args = parser.parse_args()

    global IMG_SIZE, BATCH_SIZE, EPOCHS, MODEL_SAVE_PATH
    EPOCHS = args.epochs

    dataset_dir = None
    temp_dir = None
    try:
        if args.zip:
            # User provided a zip file
            temp_dir = tempfile.mkdtemp()
            dataset_dir = extract_zip(args.zip, temp_dir)
            print(f"Using extracted dataset from zip: {dataset_dir}")
        elif args.path:
            dataset_dir = args.path
        else:
            # Fallback to folder selection dialog (if running standalone)
            from tkinter import Tk, filedialog
            root = Tk()
            root.withdraw()
            dataset_dir = filedialog.askdirectory(title="Select folder containing 'benign' and 'malignant'")
            root.destroy()
            if not dataset_dir:
                print("No folder selected. Exiting.")
                return

        X, y, le = prepare_data(dataset_dir)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
        print(f"Train: {len(X_train)}, Val: {len(X_val)}")

        train_datagen = ImageDataGenerator(
            rotation_range=20, width_shift_range=0.1, height_shift_range=0.1,
            horizontal_flip=True, zoom_range=0.1
        )
        val_datagen = ImageDataGenerator()
        train_gen = train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE)
        val_gen = val_datagen.flow(X_val, y_val, batch_size=BATCH_SIZE)

        model, base_model = create_model()
        model.compile(optimizer=Adam(0.0001), loss='binary_crossentropy', metrics=['accuracy'])
        os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
        callbacks = [
            EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss'),
            ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_accuracy', mode='max')
        ]
        print("--- Phase 1: Training frozen base ---")
        model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=callbacks, verbose=1)

        print("--- Phase 2: Fine-tuning ---")
        base_model.trainable = True
        for layer in base_model.layers[:150]:
            layer.trainable = False
        model.compile(optimizer=Adam(0.00001), loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS//2, callbacks=callbacks, verbose=1)

        model.save(MODEL_SAVE_PATH)
        val_loss, val_acc = model.evaluate(val_gen, verbose=0)
        print(f"✅ Model saved to {MODEL_SAVE_PATH}")
        print(f"✅ Final Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

if __name__ == '__main__':
    main()
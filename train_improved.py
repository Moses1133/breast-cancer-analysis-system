"""
Improved training script with validation, data augmentation, and monitoring
"""
import os
import zipfile
import shutil
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils import class_weight
import argparse

def train_model(zip_path, epochs=10, batch_size=32):
    """Train the breast cancer model with validation and monitoring"""
    
    # Extract and prepare dataset
    extract_path = 'temp_dataset'
    if os.path.exists(extract_path):
        shutil.rmtree(extract_path)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    
    benign_path = os.path.join(extract_path, 'benign')
    malignant_path = os.path.join(extract_path, 'malignant')
    
    # Check class balance
    benign_count = len([f for f in os.listdir(benign_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    malignant_count = len([f for f in os.listdir(malignant_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    print(f"\n=== Dataset Statistics ===")
    print(f"Benign images: {benign_count}")
    print(f"Malignant images: {malignant_count}")
    print(f"Ratio: {benign_count/malignant_count:.2f}:1\n")
    
    if abs(benign_count - malignant_count) > max(benign_count, malignant_count) * 0.5:
        print("WARNING: Class imbalance detected! Consider balancing your dataset.")
    
    # Data augmentation to improve generalization
    train_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2  # 80% train, 20% validation
    )
    
    # Load training data
    train_data = train_datagen.flow_from_directory(
        extract_path,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary',
        classes={'benign': 0, 'malignant': 1},
        subset='training',
        seed=42
    )
    
    # Load validation data
    val_data = train_datagen.flow_from_directory(
        extract_path,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary',
        classes={'benign': 0, 'malignant': 1},
        subset='validation',
        seed=42
    )
    
    # Build model
    model = keras.Sequential([
        layers.Input(shape=(224, 224, 3)),
        
        # Preprocessing
        layers.Rescaling(1./127.5, offset=-1),
        
        # EfficientNetB4 backbone
        keras.applications.EfficientNetB4(
            include_top=False,
            weights='imagenet',
            input_shape=(224, 224, 3)
        ),
        
        # Global pooling
        layers.GlobalAveragePooling2D(),
        
        # Dropout for regularization
        layers.Dropout(0.5),
        
        # Dense layers
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        # Output layer
        layers.Dense(1, activation='sigmoid')
    ])
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC()]
    )
    
    print(f"=== Model Summary ===")
    model.summary()
    
    # Calculate class weights to handle imbalance
    class_weights_dict = class_weight.compute_class_weight(
        'balanced',
        classes=np.array([0, 1]),
        y=np.concatenate([
            np.zeros(benign_count),
            np.ones(malignant_count)
        ])
    )
    class_weights = {0: class_weights_dict[0], 1: class_weights_dict[1]}
    print(f"\nClass weights: {class_weights}\n")
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            'models/saved/breast_cancer_model_checkpoint.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train
    print(f"Training for {epochs} epochs...")
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print("\n=== Evaluation ===")
    val_loss, val_acc, val_auc = model.evaluate(val_data)
    print(f"Validation Accuracy: {val_acc*100:.2f}%")
    print(f"Validation AUC: {val_auc:.4f}")
    print(f"Validation Loss: {val_loss:.4f}")
    
    # Save final model
    os.makedirs('models/saved', exist_ok=True)
    model.save('models/saved/breast_cancer_model.h5')
    print("\nModel saved to models/saved/breast_cancer_model.h5")
    
    # Cleanup
    shutil.rmtree(extract_path)
    
    return history

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--zip', required=True, help='Path to dataset ZIP')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    
    args = parser.parse_args()
    train_model(args.zip, epochs=args.epochs, batch_size=args.batch_size)

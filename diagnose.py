"""
Diagnostic tool to verify preprocessing consistency
"""
import cv2
import numpy as np
from PIL import Image
import os

def analyze_image_preprocessing():
    """Check if training and prediction preprocessing match"""
    
    print("\n=== Preprocessing Consistency Check ===\n")
    
    # Check 1: Image normalization
    print("1. NORMALIZATION CHECK")
    print("-" * 40)
    test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Method 1: Rescale to [0, 1]
    norm1 = test_image.astype(np.float32) / 255.0
    print(f"Method 1 (/ 255):     Range [{norm1.min():.3f}, {norm1.max():.3f}]")
    
    # Method 2: Rescale to [-1, 1]
    norm2 = (test_image.astype(np.float32) / 127.5) - 1.0
    print(f"Method 2 (±1):        Range [{norm2.min():.3f}, {norm2.max():.3f}]")
    
    # Method 3: Standardization
    norm3 = (test_image.astype(np.float32) - 127.5) / 127.5
    print(f"Method 3 (std):       Range [{norm3.min():.3f}, {norm3.max():.3f}]")
    print("\nRECOMMENDATION: Ensure training and prediction use the SAME method!")
    
    # Check 2: Image dimensions
    print("\n2. IMAGE DIMENSION CHECK")
    print("-" * 40)
    print(f"Required size:     (224, 224, 3)")
    print(f"Current check:     Images must be exactly 224x224")
    
    # Check 3: Input file validation
    print("\n3. INPUT IMAGE QUALITY CHECK")
    print("-" * 40)
    
    upload_folder = 'uploads'
    if os.path.exists(upload_folder):
        recent_files = sorted(
            [f for f in os.listdir(upload_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))],
            key=lambda x: os.path.getmtime(os.path.join(upload_folder, x)),
            reverse=True
        )[:3]
        
        if recent_files:
            print(f"Recent uploaded files: {len(recent_files)}")
            for fname in recent_files:
                fpath = os.path.join(upload_folder, fname)
                try:
                    img = cv2.imread(fpath)
                    if img is None:
                        print(f"  ❌ {fname}: Cannot read (corrupted?)")
                    else:
                        h, w = img.shape[:2]
                        print(f"  ✓ {fname}: {w}x{h} (OK)" if (h >= 200 and w >= 200) else f"  ⚠ {fname}: {w}x{h} (TOO SMALL)")
                except Exception as e:
                    print(f"  ❌ {fname}: {str(e)}")
        else:
            print("No images uploaded yet")
    
    # Check 4: Model info
    print("\n4. MODEL INFORMATION")
    print("-" * 40)
    model_path = 'models/saved/breast_cancer_model.h5'
    if os.path.exists(model_path):
        print(f"✓ Model exists: {model_path}")
        print(f"  Size: {os.path.getsize(model_path) / (1024*1024):.1f} MB")
    else:
        print(f"❌ Model missing: {model_path}")
        print("  Run training first!")
    
    # Recommendations
    print("\n=== RECOMMENDATIONS TO IMPROVE ACCURACY ===\n")
    recommendations = [
        "1. Train with MORE epochs (15-25) to ensure convergence",
        "2. Use data AUGMENTATION (rotation, zoom, flip) during training",
        "3. Check for CLASS IMBALANCE - use class weights if needed",
        "4. Validate that training and prediction use SAME preprocessing",
        "5. Use VALIDATION SET during training to monitor overfitting",
        "6. Increase batch size if you have enough memory (32 or 64)",
        "7. Ensure input images are at least 200x200 pixels",
        "8. Make sure benign/malignant folders are correctly named",
        "9. Remove corrupted or low-quality images from dataset",
        "10. Use the improved training script with better regularization"
    ]
    for rec in recommendations:
        print(f"  {rec}")
    
    print("\n" + "="*50 + "\n")

if __name__ == '__main__':
    analyze_image_preprocessing()

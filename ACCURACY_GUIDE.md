# Improving Prediction Accuracy

## Quick Diagnostics

Run this command to diagnose issues:
```bash
python diagnose.py
```

---

## Common Problems & Solutions

### 1. **Random/Inconsistent Predictions**
**Cause:** Model not properly trained or preprocessing mismatch

**Solutions:**
- Retrain with the improved script: `python train_improved.py --zip your_dataset.zip --epochs 20`
- Ensure your dataset has balanced classes (similar number of benign and malignant)
- Check image quality - remove blurry/corrupted images

### 2. **Always Predicting Same Class**
**Cause:** Severe class imbalance or poor training

**Solutions:**
- Balance your dataset - ensure roughly equal benign/malignant images
- Use the improved script which auto-calculates class weights
- Retrain with more epochs (15-25 minimum)

### 3. **Confidence Scores Always Too High**
**Cause:** Model is overconfident (overfitting)

**Solutions:**
- Use the improved script with Dropout regularization
- Add data augmentation during training
- Use EarlyStopping to prevent overfitting

### 4. **Predictions Wrong on Specific Image Types**
**Cause:** Training data doesn't match your prediction data

**Solutions:**
- Ensure training images are same quality/format as prediction images
- Use similar imaging equipment/protocols
- Include diverse image variants in training

---

## Step-by-Step Fix

### Step 1: Prepare Dataset
```
your_dataset.zip
├── benign/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...  (200+ images recommended)
└── malignant/
    ├── image1.jpg
    ├── image2.jpg
    └── ...  (200+ images recommended)
```

**Requirements:**
- Balanced classes (e.g., 500 benign, 500 malignant)
- Images at least 200x200 pixels
- JPG or PNG format
- No corrupted files

### Step 2: Run Improved Training
```bash
python train_improved.py --zip your_dataset.zip --epochs 20 --batch-size 32
```

This script will:
- ✓ Check class balance and warn if imbalanced
- ✓ Use data augmentation (rotation, zoom, flip)
- ✓ Use validation set (80/20 split)
- ✓ Calculate automatic class weights
- ✓ Use EarlyStopping to prevent overfitting
- ✓ Save checkpoints during training

### Step 3: Monitor Training
Watch the output for:
- Training accuracy should increase each epoch
- Validation accuracy should follow closely
- If validation accuracy decreases while training increases → overfitting

### Step 4: Test Predictions
Upload test images and verify accuracy improves

---

## Technical Details

### Preprocessing (MUST be consistent)
**Training (improved_train.py):**
```python
# Rescales to [-1, 1] range
layers.Rescaling(1./127.5, offset=-1)
```

**Prediction (your_app.py):**
```python
# Should use same preprocessing
img = img.astype(np.float32) / 127.5 - 1.0
```

### Model Architecture
- **Base:** EfficientNetB4 (pre-trained on ImageNet)
- **Pooling:** Global Average Pooling
- **Dense Layers:** 256 → 128 with Dropout
- **Activation:** ReLU (hidden), Sigmoid (output)
- **Loss:** Binary Crossentropy

### Training Parameters
| Parameter | Recommended | Notes |
|-----------|------------|-------|
| Epochs | 15-25 | Use EarlyStopping to prevent overfitting |
| Batch Size | 32-64 | Smaller if out of memory |
| Learning Rate | 0.001 | Auto-reduced on plateau |
| Dropout | 0.3-0.5 | Prevents overfitting |
| Data Augmentation | Yes | Essential for good generalization |

---

## Validation Metrics to Watch

After training, check these metrics:
- **Accuracy:** 85%+ is good, 90%+ is excellent
- **AUC:** 0.85+ is good, 0.95+ is excellent
- **Loss:** Should decrease and stabilize

---

## Advanced Tips

### If Still Getting Poor Results:

1. **Get More Data**
   - 1000+ images per class is ideal
   - Ensure diverse imaging conditions

2. **Improve Data Quality**
   - Remove low-quality/blurry images
   - Ensure consistent image preprocessing
   - Remove corrupted files

3. **Try Different Architectures**
   - Currently using EfficientNetB4
   - Could try ResNet50 or DenseNet121
   - Contact developer to swap model

4. **Hyperparameter Tuning**
   - Try learning rates: 0.0001, 0.0005, 0.001, 0.005
   - Try dropout: 0.2, 0.3, 0.4, 0.5
   - Increase epochs: 25, 50, 100

5. **Class Balancing**
   - Under-sample majority class
   - Over-sample minority class
   - Use class weights (automatic in improved script)

---

## Files Used

- `train_improved.py` - New training script with improvements
- `diagnose.py` - Run to check for issues
- `models/saved/breast_cancer_model.h5` - Final trained model
- `models/saved/breast_cancer_model_checkpoint.h5` - Best checkpoint

---

## Contact Support

If issues persist:
1. Run `python diagnose.py` and save output
2. Check that image preprocessing is consistent
3. Verify dataset is balanced and high quality
4. Retrain with `train_improved.py`

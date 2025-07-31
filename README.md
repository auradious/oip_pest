# Insect Classification CNN Model

This project implements a Convolutional Neural Network (CNN) for classifying insects using TensorFlow/Keras. The model follows the specified architecture:

**Input Image → Convolution → Feature Maps → Pooling → Pooled Feature Maps → Convolution → Feature Maps → Pooling → Pooled Maps → Flatten Layer → CNN → Output**

## Dataset Structure

Your dataset should be organized as follows:
```
dataset/
├── ants/
├── bees/
├── beetle/
├── catterpillar/
├── earthworms/
├── earwig/
├── grasshopper/
├── moth/
├── slug/
├── snail/
├── wasp/
└── weevil/
```

Each folder contains images of the respective insect class.

## Setup Instructions

### Method 1: Automatic Setup (Recommended)
1. Run the setup batch file:
   ```cmd
   setup.bat
   ```
   This will:
   - Activate the virtual environment
   - Install all required packages
   - Test the installation
   - Run setup verification

### Method 2: Manual Setup
1. Activate the virtual environment:
   ```cmd
   .venv\Scripts\activate
   ```

2. Install required packages:
   ```cmd
   pip install -r requirements.txt
   ```

3. Test the setup:
   ```cmd
   python test_setup.py
   ```

## Training the Model

### Option 1: Simple Training Script
```cmd
python train_model.py
```

### Option 2: Full Control
```cmd
python insect_cnn_model.py
```

### Option 3: Interactive Usage
```python
from insect_cnn_model import InsectCNNModel

# Create model instance
model = InsectCNNModel(dataset_path="dataset")

# Load data
train_ds, val_ds = model.load_and_preprocess_data()

# Create and compile model
model.create_cnn_model()
model.compile_model()

# Train
model.train_model(train_ds, val_ds, epochs=25)

# Evaluate
model.evaluate_model(val_ds)

# Save model
model.save_model("my_model.h5")
```

## Model Architecture

The CNN model includes:

1. **Data Augmentation Layer**
   - Random horizontal flip
   - Random rotation (10%)
   - Random zoom (10%)
   - Random contrast adjustment (10%)

2. **Convolution Blocks**
   - Conv1: 32 filters (3x3) + BatchNorm + MaxPool
   - Conv2: 64 filters (3x3) + BatchNorm + MaxPool
   - Conv3: 128 filters (3x3) + BatchNorm + MaxPool
   - Conv4: 256 filters (3x3) + BatchNorm + MaxPool

3. **Fully Connected Layers**
   - Flatten layer
   - Dense: 512 units + ReLU + Dropout(0.5)
   - Dense: 256 units + ReLU + Dropout(0.3)
   - Output: 12 units (softmax) for 12 insect classes

## Features

- **Data Augmentation**: Improves model generalization
- **Early Stopping**: Prevents overfitting
- **Learning Rate Scheduling**: Adaptive learning rate
- **Batch Normalization**: Faster convergence
- **Dropout**: Regularization
- **Visualization**: Training plots and confusion matrix
- **Model Persistence**: Save/load trained models
- **Single Image Prediction**: Classify individual images

## Output Files

After training, the following files will be generated:
- `insect_cnn_model.h5`: Trained model
- `training_history.png`: Training/validation accuracy and loss plots
- `confusion_matrix.png`: Classification confusion matrix

## Requirements

- Python 3.12
- TensorFlow >= 2.13.0
- NumPy >= 1.24.0
- Matplotlib >= 3.7.0
- Scikit-learn >= 1.3.0
- Seaborn >= 0.12.0
- Pillow >= 10.0.0

## Troubleshooting

1. **Import Errors**: Make sure the virtual environment is activated
2. **Dataset Errors**: Verify the dataset folder structure
3. **Memory Errors**: Reduce batch_size in the model initialization
4. **Training Slow**: Consider using GPU support if available

## Model Performance

The model includes several features to ensure good performance:
- Cross-validation with 80/20 train/validation split
- Early stopping to prevent overfitting
- Comprehensive evaluation metrics
- Visual performance analysis

## Customization

You can customize the model by modifying parameters in `InsectCNNModel`:
- `img_height`, `img_width`: Input image dimensions
- `batch_size`: Training batch size
- Learning rate in `compile_model()`
- Number of epochs in `train_model()`

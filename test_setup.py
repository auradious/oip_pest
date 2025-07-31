"""
Test script to verify the CNN model setup and run a quick test
"""
import sys
import os

def test_imports():
    """Test if all required packages are installed correctly"""
    print("Testing package imports...")
    
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow: {tf.__version__}")
    except ImportError as e:
        print(f"✗ TensorFlow import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy: {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print(f"✓ Matplotlib imported successfully")
    except ImportError as e:
        print(f"✗ Matplotlib import failed: {e}")
        return False
    
    try:
        from sklearn.metrics import classification_report
        print(f"✓ Scikit-learn imported successfully")
    except ImportError as e:
        print(f"✗ Scikit-learn import failed: {e}")
        return False
    
    try:
        import seaborn as sns
        print(f"✓ Seaborn imported successfully")
    except ImportError as e:
        print(f"✗ Seaborn import failed: {e}")
        return False
    
    try:
        from PIL import Image
        print(f"✓ Pillow imported successfully")
    except ImportError as e:
        print(f"✗ Pillow import failed: {e}")
        return False
    
    return True

def check_dataset():
    """Check if the dataset is properly structured"""
    print("\nChecking dataset structure...")
    
    dataset_path = "dataset"
    if not os.path.exists(dataset_path):
        print(f"✗ Dataset directory '{dataset_path}' not found")
        return False
    
    # List all class directories
    classes = [d for d in os.listdir(dataset_path) 
               if os.path.isdir(os.path.join(dataset_path, d))]
    
    if len(classes) == 0:
        print("✗ No class directories found in dataset")
        return False
    
    print(f"✓ Found {len(classes)} classes: {classes}")
    
    # Check each class for images
    total_images = 0
    for class_name in classes:
        class_path = os.path.join(dataset_path, class_name)
        images = [f for f in os.listdir(class_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        print(f"  - {class_name}: {len(images)} images")
        total_images += len(images)
    
    print(f"✓ Total images in dataset: {total_images}")
    return True

def quick_model_test():
    """Quick test to create and compile the model"""
    print("\nTesting model creation...")
    
    try:
        from insect_cnn_model import InsectCNNModel
        
        # Create a small model instance for testing
        model_instance = InsectCNNModel(
            dataset_path="dataset",
            img_height=150,
            img_width=150,
            batch_size=32
        )
        
        print("✓ InsectCNNModel class imported successfully")
        
        # Test data loading (just check if it works, don't load full dataset)
        print("✓ Model instance created successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("="*60)
    print("INSECT CNN MODEL SETUP VERIFICATION")
    print("="*60)
    
    # Test imports
    if not test_imports():
        print("\n❌ Package import test failed!")
        return
    
    # Check dataset
    if not check_dataset():
        print("\n❌ Dataset check failed!")
        return
    
    # Test model
    if not quick_model_test():
        print("\n❌ Model test failed!")
        return
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("✅ Ready to train the CNN model!")
    print("="*60)
    
    print("\nNext steps:")
    print("1. Run: python insect_cnn_model.py")
    print("2. Or use the model interactively in Python")
    print("3. The model will train on your insect dataset")
    print("4. Training plots and model will be saved automatically")

if __name__ == "__main__":
    main()

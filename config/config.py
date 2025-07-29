"""
Configuration file for Organic Farm Pest Management AI System
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATASET_PATH = PROJECT_ROOT / "dataset"
MODELS_PATH = PROJECT_ROOT / "models"
SRC_PATH = PROJECT_ROOT / "src"

# Create directories if they don't exist
MODELS_PATH.mkdir(exist_ok=True)

# Dataset configuration
HARMFUL_PEST_CLASSES = [
    'beetle',
    'caterpillar', 
    'earwig',
    'grasshopper',
    'moth',
    'slug',
    'snail',
    'wasp',
    'weevil'
]

# Beneficial insects to exclude from harmful pest classification
BENEFICIAL_CLASSES = [
    'ants',      # Generally beneficial for pest control
    'bees',      # Essential pollinators
    'earthworms' # Soil health improvement
]

# All available classes in dataset
ALL_CLASSES = HARMFUL_PEST_CLASSES + BENEFICIAL_CLASSES

# Model configuration
MODEL_CONFIG = {
    'input_shape': (224, 224, 3),
    'num_classes': len(HARMFUL_PEST_CLASSES),
    'batch_size': 32,
    'epochs': 25,
    'learning_rate': 0.001,
    'validation_split': 0.2,
    'test_split': 0.15,
    'random_state': 42
}

# Data augmentation parameters
AUGMENTATION_CONFIG = {
    'rotation_range': 20,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'shear_range': 0.2,
    'zoom_range': 0.2,
    'horizontal_flip': True,
    'brightness_range': [0.8, 1.2],
    'fill_mode': 'nearest'
}

# Training configuration
TRAINING_CONFIG = {
    'early_stopping_patience': 10,
    'reduce_lr_patience': 5,
    'reduce_lr_factor': 0.2,
    'min_lr': 1e-7,
    'save_best_only': True,
    'monitor': 'val_accuracy',
    'mode': 'max'
}

# Model file paths
MODEL_PATHS = {
    'best_model': MODELS_PATH / 'best_pest_classifier.h5',
    'final_model': MODELS_PATH / 'final_pest_classifier.h5',
    'model_weights': MODELS_PATH / 'model_weights.h5',
    'class_names': MODELS_PATH / 'class_names.json',
    'training_history': MODELS_PATH / 'training_history.json'
}

# Economic impact ratings for harmful pests (1-5 scale, 5 being most severe)
ECONOMIC_IMPACT = {
    'beetle': 4,      # High crop damage, leaf consumption
    'caterpillar': 5, # Very high - significant crop losses
    'earwig': 3,      # Moderate - plant and fruit damage
    'grasshopper': 5, # Very high - extensive crop damage
    'moth': 4,        # High - larvae cause substantial damage
    'slug': 4,        # High - seedling and plant damage
    'snail': 4,       # High - plant consumption
    'wasp': 3,        # Moderate - fruit damage in some cases
    'weevil': 5       # Very high - grain/seed damage
}

# Urgency levels for treatment (based on pest behavior and damage potential)
TREATMENT_URGENCY = {
    'beetle': 'Medium',      # Can be managed with regular monitoring
    'caterpillar': 'High',   # Rapid reproduction and damage
    'earwig': 'Medium',      # Nocturnal, manageable with traps
    'grasshopper': 'High',   # Can quickly devastate crops
    'moth': 'High',          # Adult laying eggs for damaging larvae
    'slug': 'Medium',        # Slow moving, but persistent damage
    'snail': 'Medium',       # Similar to slugs
    'wasp': 'Low',           # Usually beneficial, treat only if problematic
    'weevil': 'High'         # Can destroy stored crops and seeds
}

# Gradio interface configuration
GRADIO_CONFIG = {
    'title': "🌱 Organic Farm Pest Management AI",
    'description': "Upload an image of a pest to get instant identification and organic treatment recommendations",
    'theme': "soft",
    'server_port': 7860,
    'server_name': "0.0.0.0",
    'share': False,  # Set to True for public sharing
    'debug': True
}

# Image processing configuration
IMAGE_CONFIG = {
    'target_size': (224, 224),
    'color_mode': 'rgb',
    'supported_formats': ['.png', '.jpg', '.jpeg', '.bmp', '.tiff'],
    'max_file_size_mb': 10
}

print(f"✅ Configuration loaded successfully!")
print(f"📂 Dataset path: {DATASET_PATH}")
print(f"🎯 Harmful pest classes: {len(HARMFUL_PEST_CLASSES)}")
print(f"🤖 Model input shape: {MODEL_CONFIG['input_shape']}")
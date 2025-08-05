"""
Data preprocessing module for Organic Farm Pest Management AI System
Handles loading, preprocessing, and splitting of pest image dataset
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import configuration
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.config import *

class PestDataPreprocessor:
    """
    Handles all data preprocessing tasks for pest classification
    """
    
    def __init__(self, dataset_path=DATASET_PATH):
        self.dataset_path = Path(dataset_path)
        self.target_classes = ALL_CLASSES  # Changed to include ALL classes (harmful + beneficial)
        self.target_size = MODEL_CONFIG['input_shape'][:2]
        self.class_to_idx = {}
        self.idx_to_class = {}
        self.class_counts = {}
        
        print(f"🔧 Initializing Data Preprocessor...")
        print(f"📂 Dataset path: {self.dataset_path}")
        print(f"🎯 Target classes: {self.target_classes}")
        print(f"🐛 Harmful pests: {HARMFUL_PEST_CLASSES}")
        print(f"🌟 Beneficial insects: {BENEFICIAL_CLASSES}")
        
    def explore_dataset(self):
        """
        Explore and analyze the dataset structure
        """
        print(f"\n🔍 Exploring dataset structure...")
        
        # Check if dataset path exists
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path {self.dataset_path} not found!")
        
        # Get all subdirectories (pest classes)
        available_classes = [d.name for d in self.dataset_path.iterdir() if d.is_dir()]
        print(f"📁 Available classes in dataset: {available_classes}")
        
        # Filter to include ALL target classes (harmful + beneficial)
        filtered_classes = [cls for cls in available_classes if cls in self.target_classes]
        print(f"🎯 Filtered to target classes: {filtered_classes}")
        print(f"🐛 Harmful: {[cls for cls in filtered_classes if cls in HARMFUL_PEST_CLASSES]}")
        print(f"🌟 Beneficial: {[cls for cls in filtered_classes if cls in BENEFICIAL_CLASSES]}")
        
        # Count images in each target class
        total_images = 0
        
        for class_name in filtered_classes:  # Process ALL target classes
            class_path = self.dataset_path / class_name
            
            if class_path.exists():
                # Count image files
                image_files = [f for f in class_path.glob('*') 
                             if f.suffix.lower() in IMAGE_CONFIG['supported_formats']]
                
                count = len(image_files)
                self.class_counts[class_name] = count
                total_images += count
                
                # Add emoji indicator for type
                if class_name in HARMFUL_PEST_CLASSES:
                    indicator = "🐛"
                else:
                    indicator = "🌟"
                
                print(f"  📊 {indicator} {class_name}: {count} images")
            else:
                print(f"  ❌ {class_name}: Class folder not found!")
                self.class_counts[class_name] = 0
        
        print(f"\n📈 Total images: {total_images}")
        print(f"🎯 Classes for training: {len([c for c in self.class_counts.values() if c > 0])}")
        print(f"🐛 Harmful pest classes: {len([cls for cls, count in self.class_counts.items() if count > 0 and cls in HARMFUL_PEST_CLASSES])}")
        print(f"🌟 Beneficial insect classes: {len([cls for cls, count in self.class_counts.items() if count > 0 and cls in BENEFICIAL_CLASSES])}")
        
        # Check for classes with insufficient data
        min_images_threshold = 10
        insufficient_classes = [cls for cls, count in self.class_counts.items() 
                              if count < min_images_threshold]
        
        if insufficient_classes:
            print(f"⚠️  Classes with < {min_images_threshold} images: {insufficient_classes}")
        
        return self.class_counts
    
    def visualize_dataset_distribution(self):
        """
        Create visualizations of dataset distribution
        """
        if not self.class_counts:
            self.explore_dataset()
        
        # Filter out classes with zero images
        valid_counts = {k: v for k, v in self.class_counts.items() if v > 0}
        
        if not valid_counts:
            print("❌ No valid classes found for visualization!")
            return
        
        # Create distribution plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Bar plot of class distribution with color coding
        classes = list(valid_counts.keys())
        counts = list(valid_counts.values())
        
        # Color code: red for harmful, green for beneficial
        colors = ['red' if cls in HARMFUL_PEST_CLASSES else 'green' for cls in classes]
        
        bars = ax1.bar(classes, counts, color=colors, alpha=0.7)
        ax1.set_title('🐛🌟 All Insect Classes Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Insect Classes')
        ax1.set_ylabel('Number of Images')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add legend
        from matplotlib.patches import Rectangle
        legend_elements = [Rectangle((0,0),1,1, facecolor='red', alpha=0.7, label='🐛 Harmful Pests'),
                          Rectangle((0,0),1,1, facecolor='green', alpha=0.7, label='🌟 Beneficial Insects')]
        ax1.legend(handles=legend_elements)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        # Pie chart with color coding
        pie_colors = ['red' if cls in HARMFUL_PEST_CLASSES else 'green' for cls in classes]
        ax2.pie(counts, labels=classes, autopct='%1.1f%%', colors=pie_colors, startangle=90)
        ax2.set_title('📊 Class Distribution (🐛 Red = Harmful, 🌟 Green = Beneficial)', fontsize=14, fontweight='bold')
        
        # Economic impact vs. image count
        economic_data = [(cls, self.class_counts[cls], ECONOMIC_IMPACT.get(cls, 0)) 
                        for cls in classes]
        economic_df = pd.DataFrame(economic_data, columns=['Class', 'Image_Count', 'Economic_Impact'])
        
        scatter = ax3.scatter(economic_df['Image_Count'], economic_df['Economic_Impact'], 
                            c=range(len(economic_df)), cmap='viridis', s=100, alpha=0.7)
        
        for i, cls in enumerate(economic_df['Class']):
            ax3.annotate(cls, (economic_df.iloc[i]['Image_Count'], economic_df.iloc[i]['Economic_Impact']),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax3.set_xlabel('Number of Images')
        ax3.set_ylabel('Economic Impact Rating (1-5)')
        ax3.set_title('💰 Economic Impact vs. Dataset Size', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Treatment urgency distribution
        urgency_counts = {}
        for cls in classes:
            urgency = TREATMENT_URGENCY.get(cls, 'Unknown')
            urgency_counts[urgency] = urgency_counts.get(urgency, 0) + valid_counts[cls]
        
        urgency_colors = {
            'High': '#FF6B6B', 
            'Medium': '#FFA07A', 
            'Low': '#98D8C8',
            'Protect': '#90EE90',  # Light green for beneficial insects
            'Unknown': '#CCCCCC'
        }
        urgency_labels = list(urgency_counts.keys())
        urgency_values = list(urgency_counts.values())
        colors_urgency = [urgency_colors.get(label, '#CCCCCC') for label in urgency_labels]
        
        ax4.pie(urgency_values, labels=urgency_labels, autopct='%1.1f%%', 
               colors=colors_urgency, startangle=90)
        ax4.set_title('🚨 Action Priority Distribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        print(f"\n📊 Dataset Summary:")
        print(f"  • Total classes: {len(valid_counts)}")
        print(f"  • Total images: {sum(valid_counts.values())}")
        print(f"  • Average images per class: {np.mean(list(valid_counts.values())):.1f}")
        print(f"  • Min images: {min(valid_counts.values())}")
        print(f"  • Max images: {max(valid_counts.values())}")
        
    def display_sample_images(self, samples_per_class=4):
        """
        Display sample images from each class
        """
        if not self.class_counts:
            self.explore_dataset()
        
        valid_classes = [cls for cls, count in self.class_counts.items() if count > 0]
        
        if not valid_classes:
            print("❌ No valid classes found!")
            return
        
        fig, axes = plt.subplots(len(valid_classes), samples_per_class, 
                                figsize=(samples_per_class*3, len(valid_classes)*2.5))
        
        if len(valid_classes) == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('🖼️  Sample Images from Each Insect Class (🐛 Harmful | 🌟 Beneficial)', 
                     fontsize=16, fontweight='bold', y=0.98)
        
        for i, class_name in enumerate(valid_classes):
            class_path = self.dataset_path / class_name
            image_files = list(class_path.glob('*'))
            image_files = [f for f in image_files if f.suffix.lower() in IMAGE_CONFIG['supported_formats']]
            
            # Determine if harmful or beneficial
            if class_name in HARMFUL_PEST_CLASSES:
                class_type = "🐛 Harmful"
                impact_color = "red"
                impact_value = ECONOMIC_IMPACT.get(class_name, "N/A")
            else:
                class_type = "🌟 Beneficial"
                impact_color = "green"
                impact_value = abs(ECONOMIC_IMPACT.get(class_name, 0))  # Show positive value for display
            
            for j in range(samples_per_class):
                if j < len(image_files):
                    try:
                        img = Image.open(image_files[j])
                        img = img.convert('RGB')  # Ensure RGB format
                        
                        axes[i, j].imshow(img)
                        axes[i, j].set_title(f'{class_name}\n{class_type}\nImpact: {impact_value}/5', 
                                           fontsize=10, fontweight='bold', color=impact_color)
                        axes[i, j].axis('off')
                        
                    except Exception as e:
                        axes[i, j].text(0.5, 0.5, f'Error\nloading\nimage\n{str(e)[:20]}...', 
                                       ha='center', va='center', transform=axes[i, j].transAxes,
                                       fontsize=8)
                        axes[i, j].axis('off')
                else:
                    axes[i, j].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def load_and_preprocess_images(self):
        """
        Load and preprocess all images for training
        """
        print(f"\n🔄 Loading and preprocessing images...")
        
        if not self.class_counts:
            self.explore_dataset()
        
        # Create class mappings
        valid_classes = [cls for cls, count in self.class_counts.items() if count > 0]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(valid_classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        print(f"🏷️  Class mappings: {self.class_to_idx}")
        
        # Load images and labels
        images = []
        labels = []
        image_paths = []
        
        for class_name in valid_classes:
            class_path = self.dataset_path / class_name
            class_idx = self.class_to_idx[class_name]
            
            print(f"📂 Loading {class_name} images...")
            
            image_files = [f for f in class_path.glob('*') 
                          if f.suffix.lower() in IMAGE_CONFIG['supported_formats']]
            
            for img_path in image_files:
                try:
                    # Load and preprocess image
                    img = load_img(img_path, target_size=self.target_size)
                    img_array = img_to_array(img)
                    img_array = img_array / 255.0  # Normalize to [0,1]

                    images.append(img_array)
                    labels.append(class_idx)
                    image_paths.append(str(img_path))

                except Exception as e:
                    print(f"❌ Error loading {img_path}: {e}")

        
        # Convert to numpy arrays
        X = np.array(images)
        y = np.array(labels)
        
        print(f"✅ Loaded {len(X)} images successfully!")
        print(f"📊 Data shape: {X.shape}")
        print(f"🏷️  Labels shape: {y.shape}")
        print(f"🎯 Number of classes: {len(valid_classes)}")
        
        # Validate label range
        unique_labels = np.unique(y)
        print(f"🔍 Unique labels found: {unique_labels}")
        print(f"🔍 Expected labels range: [0, {len(valid_classes)-1}]")
        
        if len(unique_labels) != len(valid_classes):
            print(f"⚠️  WARNING: Found {len(unique_labels)} unique labels but expected {len(valid_classes)} classes!")
        
        if np.max(y) >= len(valid_classes):
            print(f"❌ ERROR: Maximum label {np.max(y)} is outside expected range [0, {len(valid_classes)-1}]!")
            raise ValueError(f"Label values exceed expected class count. Check class filtering.")
        
        # Save class mappings
        self._save_class_mappings()
        
        return X, y, image_paths
    
    def split_dataset(self, X, y, image_paths):
        """
        Split dataset into train, validation, and test sets
        """
        print(f"\n✂️  Splitting dataset...")
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test, paths_temp, paths_test = train_test_split(
            X, y, image_paths, 
            test_size=MODEL_CONFIG['test_split'],
            random_state=MODEL_CONFIG['random_state'],
            stratify=y
        )
        
        # Second split: separate train and validation from remaining data
        val_size_adjusted = MODEL_CONFIG['validation_split'] / (1 - MODEL_CONFIG['test_split'])
        X_train, X_val, y_train, y_val, paths_train, paths_val = train_test_split(
            X_temp, y_temp, paths_temp,
            test_size=val_size_adjusted,
            random_state=MODEL_CONFIG['random_state'],
            stratify=y_temp
        )
        
        # Print exact numbers
        total_samples = len(X)
        train_samples = X_train.shape[0]
        val_samples = X_val.shape[0]
        test_samples = X_test.shape[0]
        
        print(f"📊 Dataset split completed:")
        print(f"  📚 Total samples: {total_samples}")
        print(f"  🏋️  Training set: {train_samples} samples ({train_samples/total_samples*100:.1f}%)")
        print(f"  ✅ Validation set: {val_samples} samples ({val_samples/total_samples*100:.1f}%)")
        print(f"  🧪 Test set: {test_samples} samples ({test_samples/total_samples*100:.1f}%)")
        
        # Verify the split adds up correctly
        assert train_samples + val_samples + test_samples == total_samples, "Split doesn't add up!"
        print(f"  ✅ Verification: {train_samples} + {val_samples} + {test_samples} = {total_samples} ✓")
        
        # Check class distribution in splits
        self._check_split_distribution(y_train, y_val, y_test)
        
        return (X_train, X_val, X_test), (y_train, y_val, y_test), (paths_train, paths_val, paths_test)
    
    def _check_split_distribution(self, y_train, y_val, y_test):
        """
        Check and visualize class distribution across splits
        """
        splits_data = {
            'Training': y_train,
            'Validation': y_val, 
            'Test': y_test
        }
        
        print(f"\n📊 Class distribution across splits:")
        
        for split_name, y_split in splits_data.items():
            unique, counts = np.unique(y_split, return_counts=True)
            print(f"\n{split_name} set:")
            for idx, count in zip(unique, counts):
                class_name = self.idx_to_class[idx]
                percentage = count / len(y_split) * 100
                print(f"  {class_name}: {count} ({percentage:.1f}%)")
    
    def compute_class_weights(self, y_train):
        """
        Compute class weights to handle class imbalance
        """
        print(f"\n⚖️  Computing class weights...")
        
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
        
        print("📊 Class weights:")
        for idx, weight in class_weight_dict.items():
            class_name = self.idx_to_class[idx]
            print(f"  {class_name}: {weight:.3f}")
        
        return class_weight_dict
    
    def create_data_generators(self, X_train, y_train, X_val, y_val):
        """
        Create data generators with augmentation for training
        """
        print(f"\n🔄 Creating data generators...")
        
        # Training data generator with augmentation
        train_datagen = ImageDataGenerator(**AUGMENTATION_CONFIG)
        
        # Validation data generator (no augmentation)
        val_datagen = ImageDataGenerator()
        
        # Create generators
        train_generator = train_datagen.flow(
            X_train, y_train,
            batch_size=MODEL_CONFIG['batch_size'],
            shuffle=True
        )
        
        val_generator = val_datagen.flow(
            X_val, y_val,
            batch_size=MODEL_CONFIG['batch_size'],
            shuffle=False
        )
        
        print(f"✅ Data generators created successfully!")
        print(f"📊 Training batches per epoch: {len(train_generator)}")
        print(f"📊 Validation batches per epoch: {len(val_generator)}")
        
        return train_generator, val_generator
    
    def _save_class_mappings(self):
        """
        Save class mappings to JSON file
        """
        mappings = {
            'class_to_idx': self.class_to_idx,
            'idx_to_class': self.idx_to_class,
            'target_classes': self.target_classes,  # All classes (harmful + beneficial)
            'harmful_classes': HARMFUL_PEST_CLASSES,
            'beneficial_classes': BENEFICIAL_CLASSES,
            'class_counts': self.class_counts
        }
        
        with open(MODEL_PATHS['class_names'], 'w') as f:
            json.dump(mappings, f, indent=2)
        
        print(f"💾 Class mappings saved to {MODEL_PATHS['class_names']}")
    
    def get_preprocessing_summary(self):
        """
        Get summary of preprocessing steps
        """
        summary = {
            'dataset_path': str(self.dataset_path),
            'target_classes': self.target_classes,
            'harmful_classes': HARMFUL_PEST_CLASSES,
            'beneficial_classes': BENEFICIAL_CLASSES,
            'class_counts': self.class_counts,
            'target_size': self.target_size,
            'total_images': sum(self.class_counts.values()),
            'num_classes': len([c for c in self.class_counts.values() if c > 0]),
            'num_harmful': len([cls for cls, count in self.class_counts.items() if count > 0 and cls in HARMFUL_PEST_CLASSES]),
            'num_beneficial': len([cls for cls, count in self.class_counts.items() if count > 0 and cls in BENEFICIAL_CLASSES])
        }
        
        return summary

# Example usage and testing
if __name__ == "__main__":
    print("🚀 Testing Data Preprocessing Module...")
    
    # Initialize preprocessor
    preprocessor = PestDataPreprocessor()
    
    # Explore dataset
    class_counts = preprocessor.explore_dataset()
    
    # Visualize dataset
    preprocessor.visualize_dataset_distribution()
    
    # Display sample images
    preprocessor.display_sample_images()
    
    # Load and preprocess images
    X, y, image_paths = preprocessor.load_and_preprocess_images()
    
    # Split dataset
    (X_train, X_val, X_test), (y_train, y_val, y_test), _ = preprocessor.split_dataset(X, y, image_paths)
    
    # Compute class weights
    class_weights = preprocessor.compute_class_weights(y_train)
    
    # Create data generators
    train_gen, val_gen = preprocessor.create_data_generators(X_train, y_train, X_val, y_val)
    
    # Get summary
    summary = preprocessor.get_preprocessing_summary()
    print(f"\n📋 Preprocessing Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    print(f"\n✅ Data preprocessing module test completed successfully!")
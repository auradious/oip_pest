"""
OPTIMIZED CNN - Full Compatibility Fix
Works with all TensorFlow versions (2.0+)
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config.config import *
from src.data_preprocessing import PestDataPreprocessor

class CompatibleOptimizedCNN:
    """
    Optimized CNN - Compatible with all TensorFlow versions
    """
    
    def __init__(self):
        self.model = None
        self.history = None
        self.preprocessor = PestDataPreprocessor()
        self.class_names = []
        
        np.random.seed(42)
        tf.random.set_seed(42)
        
        print("🚀 COMPATIBLE OPTIMIZED CNN")
        print(f"📦 TensorFlow version: {tf.__version__}")
        print("✅ Works with all TF 2.x versions")
    
    def apply_z_score_normalization(self, X):
        """
        Apply the proven Z-score normalization
        """
        X = np.array(X, dtype=np.float32)
        
        # Per-channel Z-score normalization
        mean = np.mean(X, axis=(0, 1, 2), keepdims=True)
        std = np.std(X, axis=(0, 1, 2), keepdims=True)
        X_normalized = (X - mean) / (std + 1e-8)
        
        print(f"📊 Z-score normalization: [{X_normalized.min():.3f}, {X_normalized.max():.3f}]")
        return X_normalized
    
    def create_compatible_model(self, num_classes):
        """
        Create optimized model compatible with all TF versions
        """
        print(f"\n🏗️  Creating COMPATIBLE optimized model...")
        
        model = models.Sequential([
            layers.Input(shape=(224, 224, 3)),
            
            # Block 1: Initial feature extraction
            layers.Conv2D(32, (7, 7), activation='relu', padding='same'),
            layers.Conv2D(32, (5, 5), activation='relu', padding='same'),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((3, 3)),
            layers.Dropout(0.15),
            
            # Block 2: Deeper features
            layers.Conv2D(64, (5, 5), activation='relu', padding='same'),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 3: Complex patterns
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.3),
            
            # Block 4: High-level features
            layers.Conv2D(256, (5, 5), activation='relu', padding='same'),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.4),
            
            # Global pooling and classification
            layers.GlobalAveragePooling2D(),
            
            # Dense layers
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.4),
            
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            
            layers.Dense(num_classes, activation='softmax')
        ])
        
        # Compatible compilation - only standard metrics
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']  # Only standard accuracy for compatibility
        )
        
        print(f"✅ Compatible optimized model created!")
        print(f"📊 Parameters: {model.count_params():,}")
        
        return model
    
    def create_compatible_augmentation(self):
        """
        Create data augmentation compatible with older TF versions
        """
        # Use ImageDataGenerator for maximum compatibility
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        
        return ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            brightness_range=[0.9, 1.1],
            fill_mode='nearest'
        )
    
    def run_compatible_training(self):
        """
        Run training with full compatibility
        """
        print(f"\n🚀 STARTING COMPATIBLE TRAINING...")
        
        # Load and preprocess data
        print("📁 Loading data...")
        class_counts = self.preprocessor.explore_dataset()
        X, y, image_paths = self.preprocessor.load_and_preprocess_images()
        
        # Apply proven normalization
        X = self.apply_z_score_normalization(X)
        y = np.array(y, dtype=np.int32)
        
        # Split data
        (X_train, X_val, X_test), (y_train, y_val, y_test), _ = self.preprocessor.split_dataset(X, y, image_paths)
        
        # Convert to numpy
        X_train = np.array(X_train)
        X_val = np.array(X_val)
        X_test = np.array(X_test)
        y_train = np.array(y_train)
        y_val = np.array(y_val)
        y_test = np.array(y_test)
        
        # Get class info
        num_classes = len(np.unique(y))
        self.class_names = [self.preprocessor.idx_to_class[i] for i in range(num_classes)]
        
        print(f"\n🎯 Dataset: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test")
        print(f"🎯 Classes: {num_classes}")
        
        # Create model
        self.model = self.create_compatible_model(num_classes)

        # Test model BEFORE training
        print(f"\n🧪 Testing model compilation...")
        test_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
        test_output = self.model.predict(test_input, verbose=0)
        print(f"Model test input shape: {test_input.shape}")
        print(f"Model test output shape: {test_output.shape}")
        print(f"Model expects {self.model.input_shape} -> outputs {self.model.output_shape}")
        
        # Create data augmentation
        datagen = self.create_compatible_augmentation()
        
        print(f"\n📋 Model Summary:")
        self.model.summary()
        
        # Compatible callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,
                patience=7,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Training with data augmentation
        print(f"\n🚀 Training with data augmentation...")
        print(f"🎯 Target: >60% accuracy!")
        
        # Use fit_generator for compatibility
        self.history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=32),
            steps_per_epoch=len(X_train) // 32,
            epochs=150,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # Final evaluation
        print(f"\n📊 Final Evaluation:")
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        self.model.save_weights('./models/best.weights.h5')
        
        print(f"🎯 FINAL TEST ACCURACY: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        
        # Calculate top-3 accuracy manually for compatibility
        y_pred_proba = self.model.predict(X_test, verbose=0)
        top3_accuracy = self.calculate_top3_accuracy(y_test, y_pred_proba)
        print(f"🎯 TOP-3 ACCURACY: {top3_accuracy:.4f} ({top3_accuracy*100:.2f}%)")
        
        # Compare improvements
        improvement = test_accuracy / 0.4489  # Compare to previous 44.89%
        print(f"🚀 IMPROVEMENT over previous: {improvement:.2f}x")
        
        if test_accuracy > 0.60:
            print(f"🏆 EXCELLENT! Target achieved!")
        elif test_accuracy > 0.55:
            print(f"🎉 VERY GOOD! Close to target!")
        elif test_accuracy > 0.50:
            print(f"✅ GOOD! Significant progress!")
        else:
            print(f"⚠️  Some improvement, can optimize further")
        
        # Detailed analysis
        self.analyze_results(X_test, y_test)
        
        return test_accuracy
    
    def calculate_top3_accuracy(self, y_true, y_pred_proba):
        """
        Manually calculate top-3 accuracy for compatibility
        """
        # Get top 3 predictions for each sample
        top3_pred = np.argsort(y_pred_proba, axis=1)[:, -3:]
        
        # Check if true label is in top 3
        correct = 0
        for i, true_label in enumerate(y_true):
            if true_label in top3_pred[i]:
                correct += 1
        
        return correct / len(y_true)
    
    def analyze_results(self, X_test, y_test):
        """
        Analyze results with compatibility
        """
        # Predictions
        y_pred_proba = self.model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Classification report
        print(f"\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=self.class_names))
        
        # Per-class analysis
        print(f"\n🔍 Per-Class Analysis:")
        for i, class_name in enumerate(self.class_names):
            class_mask = y_test == i
            if np.sum(class_mask) > 0:
                class_acc = np.sum((y_test == y_pred) & class_mask) / np.sum(class_mask)
                class_samples = np.sum(class_mask)
                
                if class_acc > 0.7:
                    status = "🏆 EXCELLENT"
                elif class_acc > 0.5:
                    status = "✅ GOOD"
                elif class_acc > 0.3:
                    status = "⚠️  FAIR"
                else:
                    status = "❌ POOR"
                
                print(f"  {class_name:12s}: {class_acc:.3f} ({class_acc*100:.1f}%) - {status} ({class_samples} samples)")
        
        # Plot results
        self.plot_results(y_test, y_pred)
    
    def plot_results(self, y_test, y_pred):
        """
        Plot results with compatibility
        """
        cm = confusion_matrix(y_test, y_pred)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Confusion matrix
        try:
            import seaborn as sns
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=self.class_names,
                       yticklabels=self.class_names, ax=ax1)
        except ImportError:
            # Fallback without seaborn
            im = ax1.imshow(cm, interpolation='nearest', cmap='Blues')
            ax1.figure.colorbar(im, ax=ax1)
            
            # Add text annotations
            thresh = cm.max() / 2.
            for i, j in np.ndindex(cm.shape):
                ax1.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
            
            ax1.set_xticks(range(len(self.class_names)))
            ax1.set_yticks(range(len(self.class_names)))
            ax1.set_xticklabels(self.class_names)
            ax1.set_yticklabels(self.class_names)
        
        ax1.set_title('🚀 Optimized Model Confusion Matrix')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')
        
        # Per-class accuracy
        class_accuracies = []
        for i in range(len(self.class_names)):
            class_mask = y_test == i
            if np.sum(class_mask) > 0:
                class_acc = np.sum((y_test == y_pred) & class_mask) / np.sum(class_mask)
                class_accuracies.append(class_acc)
            else:
                class_accuracies.append(0)
        
        colors = ['darkgreen' if acc > 0.7 else 'green' if acc > 0.5 else 'orange' if acc > 0.3 else 'red' 
                 for acc in class_accuracies]
        
        bars = ax2.bar(range(len(self.class_names)), class_accuracies, color=colors)
        ax2.set_title('📊 Per-Class Accuracy (Optimized)')
        ax2.set_xlabel('Class')
        ax2.set_ylabel('Accuracy')
        ax2.set_xticks(range(len(self.class_names)))
        ax2.set_xticklabels(self.class_names, rotation=45)
        ax2.axhline(y=0.6, color='red', linestyle='--', alpha=0.7, label='Target (60%)')
        ax2.legend()
        
        # Add value labels
        for bar, acc in zip(bars, class_accuracies):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        # Plot training curves
        self.plot_training_curves()
    
    def plot_training_curves(self):
        """
        Plot training curves
        """
        if self.history is None:
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        epochs = range(1, len(self.history.history['accuracy']) + 1)
        
        # Accuracy
        ax1.plot(epochs, self.history.history['accuracy'], 'b-', label='Training', linewidth=2)
        ax1.plot(epochs, self.history.history['val_accuracy'], 'r-', label='Validation', linewidth=2)
        ax1.set_title('📈 Training Progress')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0.6, color='green', linestyle='--', alpha=0.7, label='Target')
        
        # Loss
        ax2.plot(epochs, self.history.history['loss'], 'b-', label='Training', linewidth=2)
        ax2.plot(epochs, self.history.history['val_loss'], 'r-', label='Validation', linewidth=2)
        ax2.set_title('📉 Training Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Training summary
        best_val_acc = max(self.history.history['val_accuracy'])
        best_epoch = self.history.history['val_accuracy'].index(best_val_acc) + 1
        
        print(f"\n📊 Training Summary:")
        print(f"  Best validation accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
        print(f"  Best epoch: {best_epoch}")
        print(f"  Total epochs: {len(epochs)}")

# Main execution
if __name__ == "__main__":
    print("🚀 COMPATIBLE OPTIMIZED CNN")
    print("="*50)
    
    trainer = CompatibleOptimizedCNN()
    
    try:
        final_accuracy = trainer.run_compatible_training()
        
        print(f"\n🎉 TRAINING COMPLETED!")
        print(f"🎯 Final accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
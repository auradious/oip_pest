"""
Model training module for Organic Farm Pest Management AI System
Custom CNN architecture with layer-by-layer optimization
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import json
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import configuration and preprocessing
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.config import *
from src.data_preprocessing import PestDataPreprocessor

class PestClassificationModel:
    """
    Custom CNN model for harmful pest classification - optimized layer by layer
    """
    
    def __init__(self):
        self.model = None
        self.history = None
        self.preprocessor = PestDataPreprocessor()
        self.class_names = []
        self.training_start_time = None
        self.training_end_time = None
        
        print(f"🤖 Initializing Custom CNN Model...")
        print(f"🎯 Target classes: {len(HARMFUL_PEST_CLASSES)}")
        
        # Set random seeds for reproducibility
        np.random.seed(MODEL_CONFIG['random_state'])
        tf.random.set_seed(MODEL_CONFIG['random_state'])
        
    def create_model(self, num_classes):
        """
        Create optimized custom CNN architecture
        Starting simple and building up complexity
        """
        print(f"\n🏗️  Creating Custom CNN architecture...")
        print(f"📐 Input shape: {MODEL_CONFIG['input_shape']}")
        print(f"🎯 Number of classes: {num_classes}")
        
        model = models.Sequential([
            # Input layer
            layers.Input(shape=MODEL_CONFIG['input_shape']),
            
            # Data normalization (crucial for stability)
            layers.Rescaling(1./255.0),
            
            # Block 1: Start with more filters and proper padding
            layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_1'),
            layers.BatchNormalization(name='bn1_1'),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2'),
            layers.BatchNormalization(name='bn1_2'),
            layers.MaxPooling2D((2, 2), name='pool1'),
            layers.Dropout(0.2, name='dropout1'),
            
            # Block 2: Increase complexity
            layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1'),
            layers.BatchNormalization(name='bn2_1'),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2'),
            layers.BatchNormalization(name='bn2_2'),
            layers.MaxPooling2D((2, 2), name='pool2'),
            layers.Dropout(0.25, name='dropout2'),
            
            # Block 3: More feature maps
            layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1'),
            layers.BatchNormalization(name='bn3_1'),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2'),
            layers.BatchNormalization(name='bn3_2'),
            layers.MaxPooling2D((2, 2), name='pool3'),
            layers.Dropout(0.3, name='dropout3'),
            
            # Block 4: Deep features
            layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1'),
            layers.BatchNormalization(name='bn4_1'),
            layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2'),
            layers.BatchNormalization(name='bn4_2'),
            layers.MaxPooling2D((2, 2), name='pool4'),
            layers.Dropout(0.35, name='dropout4'),
            
            # Global pooling instead of flatten (reduces overfitting)
            layers.GlobalAveragePooling2D(name='global_avg_pool'),
            
            # Dense layers with proper regularization
            layers.Dense(512, activation='relu', name='dense1'),
            layers.BatchNormalization(name='bn_dense1'),
            layers.Dropout(0.5, name='dropout_dense1'),
            
            layers.Dense(256, activation='relu', name='dense2'),
            layers.BatchNormalization(name='bn_dense2'),
            layers.Dropout(0.4, name='dropout_dense2'),
            
            # Output layer
            layers.Dense(num_classes, activation='softmax', name='predictions')
        ])
        
        # Use a proper learning rate and optimizer
        initial_learning_rate = 0.001  # Higher than before
        
        model.compile(
            optimizer=optimizers.Adam(
                learning_rate=initial_learning_rate,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-7
            ),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'top_2_accuracy']  # Added top-2 accuracy for better monitoring
        )
        
        print(f"✅ Custom CNN model created successfully!")
        print(f"📊 Total parameters: {model.count_params():,}")
        print(f"🎯 Output classes: {num_classes}")
        print(f"📚 Learning rate: {initial_learning_rate}")
        
        return model
    
    def create_callbacks(self):
        """
        Create optimized training callbacks
        """
        print(f"\n🔧 Setting up training callbacks...")
        
        callbacks = [
            # Early stopping with more patience
            EarlyStopping(
                monitor='val_accuracy',  # Monitor accuracy instead of loss
                patience=15,  # More patience
                restore_best_weights=True,
                verbose=1,
                mode='max'  # Maximize accuracy
            ),
            
            # Learning rate reduction
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,  # Cut learning rate in half
                patience=7,  # Wait 7 epochs
                min_lr=1e-7,
                verbose=1,
                cooldown=3
            ),
            
            # Model checkpoint
            ModelCheckpoint(
                filepath=str(MODEL_PATHS['best_model']),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1,
                mode='max'
            )
        ]
        
        print(f"✅ Callbacks configured:")
        print(f"  • Early stopping patience: 15 epochs")
        print(f"  • Learning rate reduction patience: 7 epochs")
        print(f"  • Monitoring: val_accuracy (maximize)")
        
        return callbacks
    
    def train_model(self, save_model=True):
        """
        Train the custom CNN model with proper data handling
        """
        print(f"\n🚀 Starting Custom CNN training pipeline...")
        self.training_start_time = datetime.now()
        
        # Step 1: Data preprocessing
        print(f"\n1️⃣  Data Preprocessing Phase")
        
        # Explore dataset
        class_counts = self.preprocessor.explore_dataset()
        
        # Check if we have enough data
        valid_classes = [cls for cls, count in class_counts.items() if count > 0]
        if len(valid_classes) < 2:
            raise ValueError("Need at least 2 classes with data for training!")
        
        print(f"🎯 Found {len(valid_classes)} classes with data")
        
        # Load and preprocess images
        X, y, image_paths = self.preprocessor.load_and_preprocess_images()
        
        print(f"📊 Total samples loaded: {len(X)}")
        print(f"📊 Image shape: {X[0].shape}")
        print(f"📊 Label range: {min(y)} to {max(y)}")
        
        # Split dataset
        (X_train, X_val, X_test), (y_train, y_val, y_test), _ = self.preprocessor.split_dataset(X, y, image_paths)
        
        # Print actual split numbers
        print(f"\n📊 Final dataset split:")
        print(f"  🏋️  Training samples: {len(X_train)}")
        print(f"  ✅ Validation samples: {len(X_val)}")
        print(f"  🧪 Test samples: {len(X_test)}")
        
        # Check class distribution in training set
        unique, counts = np.unique(y_train, return_counts=True)
        print(f"\n📊 Training set class distribution:")
        for class_idx, count in zip(unique, counts):
            class_name = self.preprocessor.idx_to_class[class_idx]
            print(f"  {class_name}: {count} samples")
        
        # Compute class weights for imbalanced data
        class_weights = self.preprocessor.compute_class_weights(y_train)
        print(f"\n⚖️  Class weights computed for balancing")
        
        # Create data generators with better augmentation
        train_generator, val_generator = self.preprocessor.create_data_generators(
            X_train, y_train, X_val, y_val
        )
        
        # Step 2: Model creation
        print(f"\n2️⃣  Model Creation Phase")
        num_classes = len(valid_classes)
        self.model = self.create_model(num_classes)
        self.class_names = [self.preprocessor.idx_to_class[i] for i in range(num_classes)]
        
        # Display model architecture
        print(f"\n📋 Custom CNN Model Architecture:")
        self.model.summary()
        
        # Step 3: Training
        print(f"\n3️⃣  Training Phase")
        
        callbacks = self.create_callbacks()
        
        print(f"🎯 Training for up to {MODEL_CONFIG['epochs']} epochs...")
        print(f"📊 Batch size: {MODEL_CONFIG['batch_size']}")
        print(f"🔄 Steps per epoch: {len(train_generator)}")
        print(f"✅ Validation steps: {len(val_generator)}")
        
        # Train the model
        self.history = self.model.fit(
            train_generator,
            epochs=MODEL_CONFIG['epochs'],
            validation_data=val_generator,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1,
            workers=4,
            use_multiprocessing=False
        )
        
        self.training_end_time = datetime.now()
        training_duration = self.training_end_time - self.training_start_time
        
        print(f"\n✅ Training completed!")
        print(f"⏱️  Training duration: {training_duration}")
        
        # Step 4: Model evaluation
        print(f"\n4️⃣  Model Evaluation Phase")
        self.evaluate_model(X_test, y_test)
        
        # Step 5: Save model and results
        if save_model:
            print(f"\n5️⃣  Saving Model Phase")
            self.save_model_and_results()
        
        return self.history
    
    def evaluate_model(self, X_test, y_test):
        """
        Comprehensive model evaluation with detailed analysis
        """
        print(f"\n📊 Evaluating Custom CNN model performance...")
        
        # Make predictions on test set
        print(f"🔄 Making predictions on {len(X_test)} test samples...")
        y_pred_proba = self.model.predict(X_test, verbose=1)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Calculate metrics
        test_accuracy = accuracy_score(y_test, y_pred)
        print(f"\n🎯 Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        
        # Check if accuracy is still low
        if test_accuracy < 0.3:
            print(f"⚠️  WARNING: Low accuracy detected! Let's analyze the issues...")
            self.diagnose_low_accuracy(X_test, y_test, y_pred_proba)
        
        # Get prediction confidence
        confidence_scores = np.max(y_pred_proba, axis=1)
        avg_confidence = np.mean(confidence_scores)
        print(f"🔍 Average Prediction Confidence: {avg_confidence:.4f} ({avg_confidence*100:.2f}%)")
        
        # Per-class accuracy
        print(f"\n📊 Per-class accuracy:")
        for i, class_name in enumerate(self.class_names):
            class_mask = y_test == i
            if np.sum(class_mask) > 0:
                class_acc = accuracy_score(y_test[class_mask], y_pred[class_mask])
                print(f"  {class_name}: {class_acc:.4f} ({class_acc*100:.2f}%) - {np.sum(class_mask)} samples")
        
        # Classification report
        print(f"\n📋 Detailed Classification Report:")
        class_report = classification_report(
            y_test, y_pred, 
            target_names=self.class_names,
            output_dict=True
        )
        
        print(classification_report(y_test, y_pred, target_names=self.class_names))
        
        # Visualizations
        self.plot_training_history()
        self.plot_confusion_matrix(y_test, y_pred)
        self.plot_prediction_analysis(y_pred_proba, y_test, y_pred)
        
        return test_accuracy, class_report
    
    def diagnose_low_accuracy(self, X_test, y_test, y_pred_proba):
        """
        Diagnose potential issues when accuracy is low
        """
        print(f"\n🔍 DIAGNOSTIC ANALYSIS - Investigating low accuracy...")
        
        # 1. Check prediction distribution
        y_pred = np.argmax(y_pred_proba, axis=1)
        pred_unique, pred_counts = np.unique(y_pred, return_counts=True)
        
        print(f"\n1️⃣  Prediction Distribution Analysis:")
        total_preds = len(y_pred)
        for class_idx, count in zip(pred_unique, pred_counts):
            class_name = self.class_names[class_idx] if class_idx < len(self.class_names) else f"Class_{class_idx}"
            print(f"  Predicted {class_name}: {count} times ({count/total_preds*100:.1f}%)")
        
        # 2. Check if model is predicting only one class
        most_predicted_class = pred_unique[np.argmax(pred_counts)]
        max_pred_ratio = np.max(pred_counts) / total_preds
        
        if max_pred_ratio > 0.8:
            print(f"❌ ISSUE: Model predicting mostly one class ({self.class_names[most_predicted_class]}: {max_pred_ratio*100:.1f}%)")
            print(f"   This suggests the model hasn't learned to distinguish between classes.")
        
        # 3. Check confidence scores
        confidence_scores = np.max(y_pred_proba, axis=1)
        avg_confidence = np.mean(confidence_scores)
        
        print(f"\n2️⃣  Confidence Analysis:")
        print(f"  Average confidence: {avg_confidence:.4f}")
        print(f"  Confidence std: {np.std(confidence_scores):.4f}")
        
        if avg_confidence < 0.4:
            print(f"❌ ISSUE: Very low confidence scores suggest the model is uncertain about its predictions.")
        
        # 4. Check true vs predicted distribution
        true_unique, true_counts = np.unique(y_test, return_counts=True)
        
        print(f"\n3️⃣  True vs Predicted Distribution:")
        print(f"  True distribution:")
        for class_idx, count in zip(true_unique, true_counts):
            class_name = self.class_names[class_idx]
            print(f"    {class_name}: {count} samples ({count/len(y_test)*100:.1f}%)")
        
        # 5. Suggest fixes
        print(f"\n🔧 SUGGESTED FIXES:")
        print(f"1. Increase learning rate (current: try 0.01 instead of 0.001)")
        print(f"2. Reduce model complexity (fewer layers or filters)")
        print(f"3. Check data quality (are images correctly labeled?)")
        print(f"4. Increase data augmentation")
        print(f"5. Train for more epochs")
        print(f"6. Check for data leakage or preprocessing issues")
        
        # 6. Show some misclassified examples
        if len(X_test) > 0:
            self.show_misclassified_examples(X_test, y_test, y_pred, max_examples=6)
    
    def show_misclassified_examples(self, X_test, y_test, y_pred, max_examples=6):
        """
        Show examples of misclassified images for analysis
        """
        misclassified_indices = np.where(y_test != y_pred)[0]
        
        if len(misclassified_indices) == 0:
            print(f"✅ No misclassified examples found!")
            return
        
        print(f"\n4️⃣  Misclassified Examples Analysis:")
        print(f"Total misclassified: {len(misclassified_indices)}")
        
        # Show a few examples
        num_examples = min(max_examples, len(misclassified_indices))
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i in range(num_examples):
            idx = misclassified_indices[i]
            
            # Denormalize image for display
            img = X_test[idx]
            if np.max(img) <= 1.0:  # If normalized
                img = (img * 255).astype(np.uint8)
            
            axes[i].imshow(img)
            axes[i].axis('off')
            
            true_class = self.class_names[y_test[idx]]
            pred_class = self.class_names[y_pred[idx]]
            
            axes[i].set_title(f"True: {true_class}\nPred: {pred_class}", 
                            fontsize=10, color='red', fontweight='bold')
        
        # Hide unused subplots
        for i in range(num_examples, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle('🔍 Misclassified Examples - Diagnostic Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def plot_prediction_analysis(self, y_pred_proba, y_test, y_pred):
        """
        Detailed prediction analysis plots
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Confidence distribution
        confidence_scores = np.max(y_pred_proba, axis=1)
        correct_predictions = (y_pred == y_test)
        
        axes[0, 0].hist(confidence_scores, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(np.mean(confidence_scores), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(confidence_scores):.3f}')
        axes[0, 0].set_xlabel('Prediction Confidence')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('🔍 Prediction Confidence Distribution', fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Correct vs Incorrect confidence
        if np.sum(~correct_predictions) > 0:
            correct_conf = confidence_scores[correct_predictions]
            incorrect_conf = confidence_scores[~correct_predictions]
            
            axes[0, 1].hist(correct_conf, bins=20, alpha=0.7, label='Correct', 
                           color='green', edgecolor='black')
            axes[0, 1].hist(incorrect_conf, bins=20, alpha=0.7, label='Incorrect', 
                           color='red', edgecolor='black')
            axes[0, 1].set_xlabel('Prediction Confidence')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('🎯 Confidence: Correct vs Incorrect', fontweight='bold')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 1].text(0.5, 0.5, 'All predictions correct!', 
                           ha='center', va='center', fontsize=16, fontweight='bold')
            axes[0, 1].set_title('🎯 Perfect Accuracy!', fontweight='bold')
        
        # 3. Per-class accuracy
        class_accuracies = []
        class_names_with_data = []
        
        for i, class_name in enumerate(self.class_names):
            class_mask = y_test == i
            if np.sum(class_mask) > 0:
                class_acc = np.mean(y_pred[class_mask] == y_test[class_mask])
                class_accuracies.append(class_acc)
                class_names_with_data.append(class_name)
        
        if class_accuracies:
            bars = axes[1, 0].bar(range(len(class_accuracies)), class_accuracies, 
                                 color='lightcoral', alpha=0.7)
            axes[1, 0].set_xlabel('Classes')
            axes[1, 0].set_ylabel('Accuracy')
            axes[1, 0].set_title('📊 Per-Class Accuracy', fontweight='bold')
            axes[1, 0].set_xticks(range(len(class_names_with_data)))
            axes[1, 0].set_xticklabels(class_names_with_data, rotation=45, ha='right')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, acc in zip(bars, class_accuracies):
                height = bar.get_height()
                axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Prediction matrix heatmap
        pred_matrix = np.zeros((len(self.class_names), len(self.class_names)))
        for true_label, pred_label in zip(y_test, y_pred):
            pred_matrix[true_label, pred_label] += 1
        
        # Normalize by row (true labels)
        pred_matrix_norm = pred_matrix / (pred_matrix.sum(axis=1, keepdims=True) + 1e-8)
        
        im = axes[1, 1].imshow(pred_matrix_norm, cmap='Blues', aspect='auto')
        axes[1, 1].set_xlabel('Predicted Label')
        axes[1, 1].set_ylabel('True Label')
        axes[1, 1].set_title('🎯 Normalized Confusion Matrix', fontweight='bold')
        axes[1, 1].set_xticks(range(len(self.class_names)))
        axes[1, 1].set_yticks(range(len(self.class_names)))
        axes[1, 1].set_xticklabels(self.class_names, rotation=45, ha='right')
        axes[1, 1].set_yticklabels(self.class_names)
        
        # Add colorbar
        plt.colorbar(im, ax=axes[1, 1])
        
        plt.tight_layout()
        plt.show()
    
    def plot_training_history(self):
        """
        Plot training history with enhanced analysis
        """
        if self.history is None:
            print("❌ No training history available!")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.history.history['accuracy']) + 1)
        
        # Accuracy plot
        axes[0, 0].plot(epochs, self.history.history['accuracy'], 'b-', 
                       label='Training Accuracy', linewidth=2)
        axes[0, 0].plot(epochs, self.history.history['val_accuracy'], 'r-', 
                       label='Validation Accuracy', linewidth=2)
        axes[0, 0].set_title('📈 Model Accuracy', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss plot
        axes[0, 1].plot(epochs, self.history.history['loss'], 'b-', 
                       label='Training Loss', linewidth=2)
        axes[0, 1].plot(epochs, self.history.history['val_loss'], 'r-', 
                       label='Validation Loss', linewidth=2)
        axes[0, 1].set_title('📉 Model Loss', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate plot (if available)
        if 'lr' in self.history.history:
            axes[1, 0].plot(epochs, self.history.history['lr'], 'g-', linewidth=2)
            axes[1, 0].set_title('📊 Learning Rate', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].axis('off')
        
        # Training summary
        final_train_acc = self.history.history['accuracy'][-1]
        final_val_acc = self.history.history['val_accuracy'][-1]
        best_val_acc = max(self.history.history['val_accuracy'])
        final_train_loss = self.history.history['loss'][-1]
        final_val_loss = self.history.history['val_loss'][-1]
        
        # Check for overfitting
        overfitting_gap = final_train_acc - final_val_acc
        overfitting_status = "⚠️ OVERFITTING" if overfitting_gap > 0.15 else "✅ Good"
        
        summary_text = f"""
Custom CNN Training Summary:

Final Training Accuracy: {final_train_acc:.4f}
Final Validation Accuracy: {final_val_acc:.4f}
Best Validation Accuracy: {best_val_acc:.4f}

Final Training Loss: {final_train_loss:.4f}
Final Validation Loss: {final_val_loss:.4f}

Overfitting Check: {overfitting_status}
(Gap: {overfitting_gap:.4f})

Total Epochs: {len(epochs)}
Classes: {len(self.class_names)}
Parameters: {self.model.count_params():,}
        """
        
        axes[1, 1].text(0.05, 0.95, summary_text, fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7),
                        transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('📊 Training Summary', fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Print diagnosis
        print(f"\n📊 Training Diagnosis:")
        if final_val_acc < 0.3:
            print(f"❌ Low validation accuracy ({final_val_acc:.3f}) - Model not learning well")
            print(f"   Suggestions: Increase learning rate, reduce model complexity, check data")
        elif overfitting_gap > 0.15:
            print(f"⚠️  Overfitting detected (gap: {overfitting_gap:.3f})")
            print(f"   Suggestions: Add more dropout, reduce model complexity, add more data")
        else:
            print(f"✅ Training looks good! Validation accuracy: {final_val_acc:.3f}")
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """
        Plot confusion matrix with enhanced details
        """
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Raw numbers
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names, ax=ax1)
        ax1.set_title('🎯 Confusion Matrix (Counts)', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Predicted Label')
        ax1.set_ylabel('True Label')
        
        # Percentages
        sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Reds',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names, ax=ax2)
        ax2.set_title('🎯 Confusion Matrix (Percentages)', fontsize=16, fontweight='bold')
        ax2.set_xlabel('Predicted Label')
        ax2.set_ylabel('True Label')
        
        plt.tight_layout()
        plt.show()
    
    def save_model_and_results(self):
        """
        Save trained model and training results
        """
        print(f"\n💾 Saving Custom CNN model and results...")
        
        # Save the final model
        self.model.save(MODEL_PATHS['final_model'])
        print(f"✅ Model saved to: {MODEL_PATHS['final_model']}")
        
        # Save training history
        if self.history:
            history_dict = {
                'model_type': 'Custom_CNN',
                'history': self.history.history,
                'class_names': self.class_names,
                'training_start_time': self.training_start_time.isoformat(),
                'training_end_time': self.training_end_time.isoformat(),
                'model_config': MODEL_CONFIG,
                'training_config': TRAINING_CONFIG
            }
            
            with open(MODEL_PATHS['training_history'], 'w') as f:
                json.dump(history_dict, f, indent=2, default=str)
            
            print(f"✅ Training history saved to: {MODEL_PATHS['training_history']}")
        
        print(f"💾 All files saved successfully!")

# Example usage and testing
if __name__ == "__main__":
    print("🚀 Testing Custom CNN Model Training Module...")
    
    # Initialize model trainer
    trainer = PestClassificationModel()
    
    # Train the model
    try:
        history = trainer.train_model(save_model=True)
        print(f"\n🎉 Custom CNN model training completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n✅ Custom CNN model training module test completed!")
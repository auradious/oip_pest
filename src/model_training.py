"""
Model training module for Organic Farm Pest Management AI System
Handles CNN model creation, training, and evaluation
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
    CNN model for harmful pest classification
    """
    
    def __init__(self):
        self.model = None
        self.history = None
        self.preprocessor = PestDataPreprocessor()
        self.class_names = []
        self.training_start_time = None
        self.training_end_time = None
        
        print(f"🤖 Initializing Pest Classification Model...")
        print(f"🎯 Target classes: {len(HARMFUL_PEST_CLASSES)}")
        
        # Set random seeds for reproducibility
        np.random.seed(MODEL_CONFIG['random_state'])
        tf.random.set_seed(MODEL_CONFIG['random_state'])
        
    def create_model(self, num_classes):
        """
        Create CNN architecture for pest classification
        """
        print(f"\n🏗️  Creating CNN model architecture...")
        
        model = models.Sequential([
            # Input layer
            layers.Input(shape=MODEL_CONFIG['input_shape']),
            
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth Convolutional Block
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fifth Convolutional Block
            layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),
            
            # Dense layers
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            
            # Output layer
            layers.Dense(num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=MODEL_CONFIG['learning_rate']),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"✅ Model created successfully!")
        print(f"📊 Total parameters: {model.count_params():,}")
        print(f"🎯 Output classes: {num_classes}")
        
        return model
    
    def create_callbacks(self):
        """
        Create training callbacks
        """
        print(f"\n🔧 Setting up training callbacks...")
        
        callbacks = [
            EarlyStopping(
                monitor=TRAINING_CONFIG['monitor'],
                patience=TRAINING_CONFIG['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1,
                mode=TRAINING_CONFIG['mode']
            ),
            
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=TRAINING_CONFIG['reduce_lr_factor'],
                patience=TRAINING_CONFIG['reduce_lr_patience'],
                min_lr=TRAINING_CONFIG['min_lr'],
                verbose=1
            ),
            
            ModelCheckpoint(
                filepath=str(MODEL_PATHS['best_model']),
                monitor=TRAINING_CONFIG['monitor'],
                save_best_only=TRAINING_CONFIG['save_best_only'],
                verbose=1,
                mode=TRAINING_CONFIG['mode']
            )
        ]
        
        print(f"✅ Callbacks configured:")
        print(f"  • Early stopping patience: {TRAINING_CONFIG['early_stopping_patience']}")
        print(f"  • Learning rate reduction patience: {TRAINING_CONFIG['reduce_lr_patience']}")
        print(f"  • Model checkpoint: {MODEL_PATHS['best_model']}")
        
        return callbacks
    
    def train_model(self, save_model=True):
        """
        Complete training pipeline
        """
        print(f"\n🚀 Starting model training pipeline...")
        self.training_start_time = datetime.now()
        
        # Step 1: Data preprocessing
        print(f"\n1️⃣  Data Preprocessing Phase")
        
        # Explore dataset
        class_counts = self.preprocessor.explore_dataset()
        
        # Check if we have enough data
        valid_classes = [cls for cls, count in class_counts.items() if count > 0]
        if len(valid_classes) < 2:
            raise ValueError("Need at least 2 classes with data for training!")
        
        # Load and preprocess images
        X, y, image_paths = self.preprocessor.load_and_preprocess_images()
        
        # Split dataset
        (X_train, X_val, X_test), (y_train, y_val, y_test), _ = self.preprocessor.split_dataset(X, y, image_paths)
        
        # Compute class weights
        class_weights = self.preprocessor.compute_class_weights(y_train)
        
        # Create data generators
        train_generator, val_generator = self.preprocessor.create_data_generators(X_train, y_train, X_val, y_val)
        
        # Step 2: Model creation
        print(f"\n2️⃣  Model Creation Phase")
        num_classes = len(valid_classes)
        self.model = self.create_model(num_classes)
        self.class_names = [self.preprocessor.idx_to_class[i] for i in range(num_classes)]
        
        # Display model architecture
        print(f"\n📋 Model Architecture Summary:")
        self.model.summary()
        
        # Step 3: Training setup
        print(f"\n3️⃣  Training Setup Phase")
        callbacks = self.create_callbacks()
        
        # Step 4: Model training
        print(f"\n4️⃣  Model Training Phase")
        print(f"🎯 Training for up to {MODEL_CONFIG['epochs']} epochs...")
        print(f"📊 Batch size: {MODEL_CONFIG['batch_size']}")
        print(f"⚖️  Using class weights to handle imbalance")
        
        self.history = self.model.fit(
            train_generator,
            epochs=MODEL_CONFIG['epochs'],
            validation_data=val_generator,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        self.training_end_time = datetime.now()
        training_duration = self.training_end_time - self.training_start_time
        
        print(f"\n✅ Training completed!")
        print(f"⏱️  Training duration: {training_duration}")
        
        # Step 5: Model evaluation
        print(f"\n5️⃣  Model Evaluation Phase")
        self.evaluate_model(X_test, y_test)
        
        # Step 6: Save model and results
        if save_model:
            print(f"\n6️⃣  Saving Model Phase")
            self.save_model_and_results()
        
        return self.history
    
    def evaluate_model(self, X_test, y_test):
        """
        Comprehensive model evaluation
        """
        print(f"\n📊 Evaluating model performance...")
        
        # Make predictions
        y_pred_proba = self.model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Calculate accuracy
        test_accuracy = accuracy_score(y_test, y_pred)
        print(f"🎯 Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        
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
        self.plot_classification_metrics(class_report)
        
        return test_accuracy, class_report
    
    def plot_training_history(self):
        """
        Plot training history
        """
        if self.history is None:
            print("❌ No training history available!")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy plot
        axes[0, 0].plot(self.history.history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
        axes[0, 0].plot(self.history.history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
        axes[0, 0].set_title('📈 Model Accuracy', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss plot
        axes[0, 1].plot(self.history.history['loss'], 'b-', label='Training Loss', linewidth=2)
        axes[0, 1].plot(self.history.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        axes[0, 1].set_title('📉 Model Loss', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate plot (if available)
        if 'lr' in self.history.history:
            axes[1, 0].plot(self.history.history['lr'], 'g-', linewidth=2)
            axes[1, 0].set_title('📊 Learning Rate Schedule', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].axis('off')
        
        # Training summary
        final_train_acc = self.history.history['accuracy'][-1]
        final_val_acc = self.history.history['val_accuracy'][-1]
        final_train_loss = self.history.history['loss'][-1]
        final_val_loss = self.history.history['val_loss'][-1]
        best_val_acc = max(self.history.history['val_accuracy'])
        
        summary_text = f"""
Training Summary:

Final Training Accuracy: {final_train_acc:.4f}
Final Validation Accuracy: {final_val_acc:.4f}
Best Validation Accuracy: {best_val_acc:.4f}

Final Training Loss: {final_train_loss:.4f}
Final Validation Loss: {final_val_loss:.4f}

Total Epochs: {len(self.history.history['accuracy'])}
Classes: {len(self.class_names)}
        """
        
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))
        axes[1, 1].set_title('📊 Training Summary', fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """
        Plot confusion matrix
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names)
        plt.title('🎯 Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
    
    def plot_classification_metrics(self, class_report):
        """
        Plot classification metrics per class
        """
        # Extract metrics for each class
        classes = [cls for cls in class_report.keys() if cls not in ['accuracy', 'macro avg', 'weighted avg']]
        
        precision = [class_report[cls]['precision'] for cls in classes]
        recall = [class_report[cls]['recall'] for cls in classes]
        f1_score = [class_report[cls]['f1-score'] for cls in classes]
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Precision, Recall, F1-Score bar plot
        x = np.arange(len(classes))
        width = 0.25
        
        ax1.bar(x - width, precision, width, label='Precision', alpha=0.8, color='skyblue')
        ax1.bar(x, recall, width, label='Recall', alpha=0.8, color='lightcoral')
        ax1.bar(x + width, f1_score, width, label='F1-Score', alpha=0.8, color='lightgreen')
        
        ax1.set_xlabel('Classes')
        ax1.set_ylabel('Score')
        ax1.set_title('📊 Classification Metrics by Class', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(classes, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.1)
        
        # Economic impact vs F1-score
        economic_impacts = [ECONOMIC_IMPACT.get(cls, 0) for cls in classes]
        scatter = ax2.scatter(economic_impacts, f1_score, s=100, alpha=0.7, c=range(len(classes)), cmap='viridis')
        
        for i, cls in enumerate(classes):
            ax2.annotate(cls, (economic_impacts[i], f1_score[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax2.set_xlabel('Economic Impact Rating')
        ax2.set_ylabel('F1-Score')
        ax2.set_title('💰 Economic Impact vs Model Performance', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Support (number of samples) per class
        support = [class_report[cls]['support'] for cls in classes]
        bars = ax3.bar(classes, support, color='orange', alpha=0.7)
        ax3.set_xlabel('Classes')
        ax3.set_ylabel('Number of Test Samples')
        ax3.set_title('📊 Test Set Distribution', fontweight='bold')
        ax3.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        # Overall metrics
        overall_metrics = {
            'Accuracy': class_report['accuracy'],
            'Macro Avg Precision': class_report['macro avg']['precision'],
            'Macro Avg Recall': class_report['macro avg']['recall'],
            'Macro Avg F1': class_report['macro avg']['f1-score'],
            'Weighted Avg Precision': class_report['weighted avg']['precision'],
            'Weighted Avg Recall': class_report['weighted avg']['recall'],
            'Weighted Avg F1': class_report['weighted avg']['f1-score']
        }
        
        metrics_text = "Overall Performance Metrics:\n\n"
        for metric, value in overall_metrics.items():
            metrics_text += f"{metric}: {value:.4f}\n"
        
        ax4.text(0.1, 0.5, metrics_text, fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.7))
        ax4.set_title('📈 Overall Performance', fontweight='bold')
        ax4.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def save_model_and_results(self):
        """
        Save trained model and training results
        """
        print(f"\n💾 Saving model and results...")
        
        # Save the final model
        self.model.save(MODEL_PATHS['final_model'])
        print(f"✅ Model saved to: {MODEL_PATHS['final_model']}")
        
        # Save training history
        if self.history:
            history_dict = {
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
    
    def load_trained_model(self, model_path=None):
        """
        Load a pre-trained model
        """
        if model_path is None:
            if MODEL_PATHS['best_model'].exists():
                model_path = MODEL_PATHS['best_model']
            elif MODEL_PATHS['final_model'].exists():
                model_path = MODEL_PATHS['final_model']
            else:
                raise FileNotFoundError("No trained model found!")
        
        print(f"📥 Loading model from: {model_path}")
        self.model = keras.models.load_model(model_path)
        
        # Load class names if available
        if MODEL_PATHS['class_names'].exists():
            with open(MODEL_PATHS['class_names'], 'r') as f:
                class_info = json.load(f)
                self.class_names = [class_info['idx_to_class'][str(i)] 
                                   for i in range(len(class_info['idx_to_class']))]
        
        print(f"✅ Model loaded successfully!")
        print(f"🎯 Classes: {self.class_names}")
        
        return self.model

# Example usage and testing
if __name__ == "__main__":
    print("🚀 Testing Model Training Module...")
    
    # Initialize model trainer
    trainer = PestClassificationModel()
    
    # Train the model
    try:
        history = trainer.train_model(save_model=True)
        print(f"\n🎉 Model training completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        raise
    
    print(f"\n✅ Model training module test completed!")
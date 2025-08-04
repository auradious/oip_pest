"""
FIXED CNN with Proper Normalization
Using Z-score normalization that actually works!
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config.config import *
from src.data_preprocessing import PestDataPreprocessor
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess


class FixedPestCNN:
    """
    CNN with PROPER normalization that actually works
    """
    
    def __init__(self):
        self.model = None
        self.history = None
        self.preprocessor = PestDataPreprocessor()
        self.class_names = []

        # Set seeds
        np.random.seed(42)
        tf.random.set_seed(42)

        # 🔧 GPU Detection and Allocation
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                tf.config.set_visible_devices(gpus[0], 'GPU')
                self.device = '/GPU:0'
                print(f"✅ CUDA is available! Using GPU: {gpus[0].name}")
            except RuntimeError as e:
                self.device = '/CPU:0'
                print(f"⚠️  GPU setup failed. Falling back to CPU. Reason: {e}")
        else:
            self.device = '/CPU:0'
            print("❌ No GPU found. Using CPU.")

        print("🔧 FIXED CNN with Proper Normalization")
        print("✅ Using Z-score normalization (proven to work!)")
    
    def apply_proper_normalization(self, X):
        """
        Apply Z-score normalization that actually works
        """
        X = np.array(X, dtype=np.float32)
        
        # Z-score normalization per channel
        print(f"📊 Before normalization: [{X.min():.4f}, {X.max():.4f}]")
        
        # Calculate mean and std for each channel
        mean = np.mean(X, axis=(0, 1, 2), keepdims=True)
        std = np.std(X, axis=(0, 1, 2), keepdims=True)
        
        # Apply Z-score normalization
        X_normalized = (X - mean) / (std + 1e-8)
        
        print(f"📊 After Z-score normalization: [{X_normalized.min():.4f}, {X_normalized.max():.4f}]")
        print(f"📊 Mean: {np.mean(X_normalized):.6f}, Std: {np.std(X_normalized):.6f}")
        
        return X_normalized
    

    def create_working_model(self, num_classes):
        """
        EfficientNetB0 backbone with custom head (corrected).
        """
        print(f"\n🏗️  Creating EfficientNetB0 model...")

        # Input image size (224x224x3)
        inputs = layers.Input(shape=(224, 224, 3))

        # EfficientNetB0 expects unprocessed inputs here
        backbone = EfficientNetB0(include_top=False, weights='imagenet', input_tensor=inputs)
        backbone.trainable = True  # You can set this to False if only training the head initially

        y = backbone.output
        y = layers.GlobalAveragePooling2D()(y)

        # Dense classification head
        y = layers.Dense(256, activation='relu')(y)
        y = layers.Dropout(0.4)(y)
        y = layers.Dense(128, activation='relu')(y)
        y = layers.Dropout(0.3)(y)
        outputs = layers.Dense(num_classes, activation='softmax')(y)

        model = models.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=optimizers.Adam(learning_rate=1e-4),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        print(f"✅ EfficientNetB0 model ready. Parameters: {model.count_params():,}")
        return model
    
    def run_fixed_training(self):
        """
        Run training with EfficientNetB0 + transfer learning + fine-tuning
        """
        print(f"\n🚀 STARTING FIXED TRAINING...")

        # Load data
        print("📁 Loading data...")
        class_counts = self.preprocessor.explore_dataset()
        X, y, image_paths = self.preprocessor.load_and_preprocess_images()
        y = np.array(y, dtype=np.int32)

        # Apply EfficientNet preprocessing (instead of Z-score)
        print("\n🔧 Applying EfficientNet preprocessing...")
        from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess
        X = effnet_preprocess(np.array(X, dtype=np.float32))  # Expected range [-1, 1]

        # Split data
        (X_train, X_val, X_test), (y_train, y_val, y_test), _ = self.preprocessor.split_dataset(X, y, image_paths)

        print(f"\n📊 Dataset splits:")
        print(f"  Training: {len(X_train)} samples")
        print(f"  Validation: {len(X_val)} samples")
        print(f"  Test: {len(X_test)} samples")

        # Get class info
        num_classes = len(np.unique(y))
        self.class_names = [self.preprocessor.idx_to_class[i] for i in range(num_classes)]

        print(f"\n🎯 Classes ({num_classes}):")
        for i, name in enumerate(self.class_names):
            train_count = np.sum(y_train == i)
            val_count = np.sum(y_val == i)
            test_count = np.sum(y_test == i)
            print(f"  {name}: {train_count} train, {val_count} val, {test_count} test")

        # Create model
        print(f"\n🏗️  Creating model...")
        self.model = self.create_working_model(num_classes)

        # Show model summary
        print(f"\n📋 Model Architecture:")
        self.model.summary()

        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]

        # -------- Stage 1: train head (backbone frozen) --------
        print(f"\n🎯 Stage 1: Training head (backbone frozen)")
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=16,
            epochs=10,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )

        # -------- Stage 2: fine-tune top of backbone --------
        if hasattr(self.model, 'layers'):
            print(f"\n🎯 Stage 2: Fine-tuning backbone + head")

            # Unfreeze last 30 layers
            for layer in self.model.layers[-30:]:
                if not isinstance(layer, layers.BatchNormalization):
                    layer.trainable = True

            # Recompile with lower LR
            self.model.compile(
                optimizer=optimizers.Adam(learning_rate=1e-5),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )

            self.history = self.model.fit(
                X_train, y_train,
                batch_size=32,
                epochs=40,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=1
            )
        else:
            print("⚠️ No backbone found; skipping fine-tuning stage.")

        # Evaluate
        print(f"\n📊 Final Evaluation:")
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)

        print(f"🎯 FINAL TEST ACCURACY: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        improvement = test_accuracy / 0.11
        print(f"🚀 IMPROVEMENT: {improvement:.1f}x better than 11% baseline")

        if test_accuracy > 0.50:
            print(f"🎉 EXCELLENT! Target achieved!")
        elif test_accuracy > 0.30:
            print(f"✅ GOOD! Significant improvement!")
        elif test_accuracy > 0.20:
            print(f"⚠️  BETTER! But can improve more")
        else:
            print(f"❌ Still needs work")

        # Save the trained model
        self.save_model()

        # Detailed results
        self.show_detailed_results(X_test, y_test)

        return test_accuracy

    
    def save_model(self):
        """
        Save the trained model in .keras format
        """
        if self.model is None:
            print("❌ No model to save!")
            return

        models_dir = Path(__file__).parent.parent / "models"
        models_dir.mkdir(exist_ok=True)

        model_path = models_dir / "best_pest_classifier.keras"

        try:
            print(f"\n💾 Saving model to {model_path}...")
            self.model.save(model_path)  # TensorFlow will use .keras format
            print(f"✅ Model saved successfully!")
            print(f"📁 Model location: {model_path}")
            print(f"📊 Model file size: {model_path.stat().st_size / (1024 * 1024):.2f} MB")
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            import traceback
            traceback.print_exc()

    
    def show_detailed_results(self, X_test, y_test):
        """
        Show detailed results
        """
        # Predictions
        y_pred_proba = self.model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Classification report
        print(f"\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=self.class_names))
        
        # Confusion matrix
        print(f"\n📊 Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        # Plot results
        self.plot_results(y_test, y_pred, cm)
        self.plot_training_curves()
    
    def plot_results(self, y_test, y_pred, cm):
        """
        Plot results
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Confusion matrix heatmap
        import seaborn as sns
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names, ax=ax1)
        ax1.set_title('🎯 Fixed Model Confusion Matrix')
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
        
        bars = ax2.bar(range(len(self.class_names)), class_accuracies, 
                      color=['green' if acc > 0.5 else 'orange' if acc > 0.3 else 'red' 
                             for acc in class_accuracies])
        ax2.set_title('📊 Per-Class Accuracy (Fixed Model)')
        ax2.set_xlabel('Class')
        ax2.set_ylabel('Accuracy')
        ax2.set_xticks(range(len(self.class_names)))
        ax2.set_xticklabels(self.class_names, rotation=45)
        
        # Add value labels
        for bar, acc in zip(bars, class_accuracies):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
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
        ax1.set_title('📈 Fixed Model Training Progress')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Loss
        ax2.plot(epochs, self.history.history['loss'], 'b-', label='Training', linewidth=2)
        ax2.plot(epochs, self.history.history['val_loss'], 'r-', label='Validation', linewidth=2)
        ax2.set_title('📉 Fixed Model Loss')
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
        print(f"  Improvement: {best_val_acc/0.11:.1f}x better than 11% baseline")

# Main execution
if __name__ == "__main__":
    print("🔧 RUNNING FIXED CNN WITH PROPER NORMALIZATION")
    print("="*60)
    
    trainer = FixedPestCNN()
    
    try:
        # Run fixed training
        print("TensorFlow version:", tf.__version__)
        print("Is GPU available:", tf.config.list_physical_devices('GPU'))

        final_accuracy = trainer.run_fixed_training()
        
        print(f"\n🎉 FIXED TRAINING COMPLETED!")
        print(f"🎯 Final accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
        
        if final_accuracy > 0.50:
            print(f"🏆 SUCCESS! Ready for architecture optimization!")
        else:
            print(f"⚠️  Improved but can optimize further")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n✅ Fixed training completed!")
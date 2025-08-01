"""
HYBRID MOBILENET CNN - Transfer Learning + Custom Architecture
Using MobileNetV2 (91% ImageNet accuracy) + optimized custom top
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
from tensorflow.keras.applications import MobileNetV2  # Changed to MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess


class HybridMobileNetCNN:
    """
    Hybrid MobileNetV2 + Custom CNN for Pest Classification
    """
    
    def __init__(self):
        self.model = None
        self.backbone = None  # Store backbone reference
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

        print("🚀 HYBRID MOBILENET + CUSTOM CNN")
        print("📱 Using MobileNetV2 (91% ImageNet accuracy) + custom pest-specific layers")
    
    def apply_proper_normalization(self, X):
        """
        Apply light normalization (MobileNet preprocessing will handle the main normalization)
        """
        X = np.array(X, dtype=np.float32)
        
        # Just ensure proper range for MobileNet preprocessing
        print(f"📊 Input range before MobileNet preprocessing: [{X.min():.4f}, {X.max():.4f}]")
        
        # MobileNet expects [0, 255] range, so ensure we have that
        if X.max() <= 1.0:
            X = X * 255.0
            print(f"📊 Scaled to [0,255] range: [{X.min():.4f}, {X.max():.4f}]")
        
        return X
    
    def create_hybrid_mobilenet_model(self, num_classes):
        """
        Create Hybrid MobileNetV2 + Custom CNN model
        """
        print(f"\n🏗️ Creating Hybrid MobileNet + Custom CNN model...")

        # Input layer
        inputs = layers.Input(shape=(224, 224, 3))
        
        # MobileNet preprocessing (converts [0,255] to [-1,1])
        x = mobilenet_preprocess(inputs)

        # MobileNetV2 backbone (91% ImageNet accuracy!)
        self.backbone = MobileNetV2(
            include_top=False, 
            weights='imagenet', 
            input_tensor=x,
            alpha=1.0,  # Width multiplier
            pooling=None  # We'll add our own pooling
        )
        
        print(f"📱 MobileNetV2 backbone loaded with ImageNet weights")
        
        # Initially freeze backbone for stage 1 training
        self.backbone.trainable = False
        
        # Get MobileNet features
        backbone_features = self.backbone.output
        
        # === CUSTOM HYBRID LAYERS ON TOP ===
        print(f"🎨 Adding custom hybrid layers for pest-specific features...")
        
        # Additional Conv layers for pest-specific feature extraction
        y = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='pest_conv1')(backbone_features)
        y = layers.BatchNormalization(name='pest_bn1')(y)
        y = layers.Dropout(0.25, name='pest_dropout1')(y)
        
        y = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='pest_conv2')(y)
        y = layers.BatchNormalization(name='pest_bn2')(y)
        y = layers.Dropout(0.25, name='pest_dropout2')(y)
        
        # Attention mechanism for important features
        attention = layers.Conv2D(256, (1, 1), activation='sigmoid', name='attention_gate')(y)
        y = layers.Multiply(name='attention_applied')([y, attention])
        
        # Multi-scale pooling for different sized insects
        # Global Average Pooling
        gap = layers.GlobalAveragePooling2D(name='global_avg_pool')(y)
        
        # Global Max Pooling 
        gmp = layers.GlobalMaxPooling2D(name='global_max_pool')(y)
        
        # Combine both pooling strategies
        y = layers.Concatenate(name='combined_pooling')([gap, gmp])
        
        # === CUSTOM DENSE HEAD ===
        print(f"🧠 Adding intelligent dense layers...")
        
        # Main classification path
        y = layers.Dense(512, activation='relu', name='pest_dense1')(y)
        y = layers.BatchNormalization(name='pest_bn3')(y)
        y = layers.Dropout(0.5, name='pest_dropout3')(y)
        
        y = layers.Dense(256, activation='relu', name='pest_dense2')(y)
        y = layers.BatchNormalization(name='pest_bn4')(y)
        y = layers.Dropout(0.4, name='pest_dropout4')(y)
        
        # Specialized layers for different pest characteristics
        # Size/shape features
        size_branch = layers.Dense(128, activation='relu', name='size_features')(y)
        size_branch = layers.Dropout(0.3, name='size_dropout')(size_branch)
        
        # Color/texture features  
        texture_branch = layers.Dense(128, activation='relu', name='texture_features')(y)
        texture_branch = layers.Dropout(0.3, name='texture_dropout')(texture_branch)
        
        # Combine specialized features
        combined_features = layers.Concatenate(name='combined_features')([size_branch, texture_branch])
        
        # Final classification layer
        y = layers.Dense(128, activation='relu', name='final_dense')(combined_features)
        y = layers.Dropout(0.3, name='final_dropout')(y)
        
        # Output layer
        outputs = layers.Dense(num_classes, activation='softmax', name='pest_predictions')(y)

        # Create hybrid model
        model = models.Model(inputs=inputs, outputs=outputs, name='HybridMobileNet_PestClassifier')
        
        # Initial compilation for stage 1 (backbone frozen)
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),  # Higher LR for new layers
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        print(f"✅ Hybrid MobileNet + Custom CNN created!")
        print(f"📊 Total parameters: {model.count_params():,}")
        
        # Count trainable parameters
        trainable_params = sum([w.shape.num_elements() for w in model.trainable_weights])
        print(f"📊 Trainable parameters: {trainable_params:,}")
        print(f"📊 MobileNet frozen parameters: {self.backbone.count_params():,}")
        
        return model
    
    def run_hybrid_training(self):
        """
        Run two-stage hybrid training: 
        Stage 1: Train custom layers (MobileNet frozen)
        Stage 2: Fine-tune entire model
        """
        print(f"\n🚀 STARTING HYBRID MOBILENET TRAINING...")

        # Load data
        print("📁 Loading data...")
        class_counts = self.preprocessor.explore_dataset()
        X, y, image_paths = self.preprocessor.load_and_preprocess_images()

        # Apply proper preprocessing for MobileNet
        X = self.apply_proper_normalization(X)
        y = np.array(y, dtype=np.int32)

        # Split data
        (X_train, X_val, X_test), (y_train, y_val, y_test), _ = self.preprocessor.split_dataset(X, y, image_paths)

        # Convert to numpy arrays
        X_train = np.array(X_train)
        X_val = np.array(X_val)
        X_test = np.array(X_test)
        y_train = np.array(y_train)
        y_val = np.array(y_val)
        y_test = np.array(y_test)

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

        # Create hybrid model
        print(f"\n🏗️ Creating hybrid model...")
        self.model = self.create_hybrid_mobilenet_model(num_classes)

        # Show model summary
        print(f"\n📋 Hybrid Model Architecture:")
        self.model.summary()

        # Enhanced callbacks
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
                patience=8,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                './models/best_hybrid_mobilenet.weights.h5',
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=True,
                verbose=1
            )
        ]

        # ===== STAGE 1: Train custom layers (MobileNet frozen) =====
        print(f"\n🎯 STAGE 1: Training custom layers (MobileNet backbone frozen)")
        print(f"🔒 MobileNet frozen, training {sum([w.shape.num_elements() for w in self.model.trainable_weights]):,} parameters")
        
        stage1_history = self.model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=30,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )

        # ===== STAGE 2: Fine-tune entire model =====
        print(f"\n🎯 STAGE 2: Fine-tuning entire hybrid model")
        
        # Unfreeze MobileNet for fine-tuning
        self.backbone.trainable = True
        
        # Fine-tune only the last layers of MobileNet
        for layer in self.backbone.layers[:-20]:
            layer.trainable = False
        
        print(f"🔓 Unfrozen last 20 MobileNet layers for fine-tuning")

        # Recompile with much lower learning rate for fine-tuning
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=0.0001),  # Much lower LR
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        trainable_params = sum([w.shape.num_elements() for w in self.model.trainable_weights])
        print(f"🔧 Fine-tuning {trainable_params:,} parameters")

        stage2_history = self.model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=50,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )

        # Combine training histories
        self.history = self.combine_histories(stage1_history, stage2_history)

        # ===== FINAL EVALUATION =====
        print(f"\n📊 Final Evaluation:")
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)

        print(f"🎯 FINAL TEST ACCURACY: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

        # Calculate top-3 accuracy
        y_pred_proba = self.model.predict(X_test, verbose=0)
        top3_accuracy = self.calculate_top3_accuracy(y_test, y_pred_proba)
        print(f"🎯 TOP-3 ACCURACY: {top3_accuracy:.4f} ({top3_accuracy*100:.2f}%)")

        # Compare with previous results
        previous_best = 0.5751  # Your previous best
        improvement = test_accuracy / previous_best
        print(f"🚀 IMPROVEMENT over previous: {improvement:.2f}x")

        if test_accuracy > 0.75:
            print(f"🏆 OUTSTANDING! Hybrid model is excellent!")
        elif test_accuracy > 0.70:
            print(f"🎉 EXCELLENT! Strong hybrid performance!")
        elif test_accuracy > 0.65:
            print(f"✅ VERY GOOD! Hybrid approach working!")
        elif test_accuracy > 0.60:
            print(f"✅ GOOD! Solid improvement with hybrid!")
        else:
            print(f"⚠️ Can optimize hybrid architecture further")

        # Save final model
        self.model.save_weights('./models/final_hybrid_mobilenet.weights.h5')
        print(f"💾 Hybrid model saved!")

        # Detailed analysis
        self.show_detailed_results(X_test, y_test)

        return test_accuracy
    
    def combine_histories(self, hist1, hist2):
        """
        Combine training histories from two stages
        """
        combined = {
            'accuracy': hist1.history['accuracy'] + hist2.history['accuracy'],
            'val_accuracy': hist1.history['val_accuracy'] + hist2.history['val_accuracy'],
            'loss': hist1.history['loss'] + hist2.history['loss'],
            'val_loss': hist1.history['val_loss'] + hist2.history['val_loss']
        }
        
        class MockHistory:
            def __init__(self, history_dict):
                self.history = history_dict
        
        return MockHistory(combined)
    
    def calculate_top3_accuracy(self, y_true, y_pred_proba):
        """
        Calculate top-3 accuracy
        """
        top3_pred = np.argsort(y_pred_proba, axis=1)[:, -3:]
        correct = 0
        for i, true_label in enumerate(y_true):
            if true_label in top3_pred[i]:
                correct += 1
        return correct / len(y_true)
    
    def show_detailed_results(self, X_test, y_test):
        """
        Show detailed results with hybrid model analysis
        """
        # Predictions
        y_pred_proba = self.model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Classification report
        print(f"\n📋 Hybrid Model Classification Report:")
        print(classification_report(y_test, y_pred, target_names=self.class_names))
        
        # Per-class analysis with improvements
        print(f"\n🔍 Per-Class Analysis (Hybrid MobileNet):")
        previous_results = {
            'beetle': 0.145,
            'catterpillar': 0.062,  
            'earwig': 0.586,
            'grasshopper': 0.479,
            'moth': 0.730,
            'slug': 0.424,
            'snail': 0.733,
            'wasp': 0.933,
            'weevil': 0.918
        }
        
        for i, class_name in enumerate(self.class_names):
            class_mask = y_test == i
            if np.sum(class_mask) > 0:
                class_acc = np.sum((y_test == y_pred) & class_mask) / np.sum(class_mask)
                class_samples = np.sum(class_mask)
                
                # Calculate improvement if we have previous data
                if class_name in previous_results:
                    improvement = class_acc / previous_results[class_name]
                    improvement_text = f" ({improvement:.1f}x improvement)"
                else:
                    improvement_text = ""
                
                if class_acc > 0.8:
                    status = "🏆 OUTSTANDING"
                elif class_acc > 0.7:
                    status = "🎉 EXCELLENT"
                elif class_acc > 0.6:
                    status = "✅ VERY GOOD"
                elif class_acc > 0.5:
                    status = "✅ GOOD"
                elif class_acc > 0.3:
                    status = "⚠️  FAIR"
                else:
                    status = "❌ POOR"
                
                print(f"  {class_name:12s}: {class_acc:.3f} ({class_acc*100:.1f}%) - {status}{improvement_text} ({class_samples} samples)")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n📊 Confusion Matrix:")
        print(cm)
        
        # Plot results
        self.plot_results(y_test, y_pred, cm)
        self.plot_training_curves()
    
    def plot_results(self, y_test, y_pred, cm):
        """
        Plot hybrid model results
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Confusion matrix heatmap
        try:
            import seaborn as sns
            sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                       xticklabels=self.class_names,
                       yticklabels=self.class_names, ax=ax1)
        except ImportError:
            # Fallback without seaborn
            im = ax1.imshow(cm, interpolation='nearest', cmap='Greens')
            ax1.figure.colorbar(im, ax=ax1)
            
            thresh = cm.max() / 2.
            for i, j in np.ndindex(cm.shape):
                ax1.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
            
            ax1.set_xticks(range(len(self.class_names)))
            ax1.set_yticks(range(len(self.class_names)))
            ax1.set_xticklabels(self.class_names)
            ax1.set_yticklabels(self.class_names)
            
        ax1.set_title('🚀 Hybrid MobileNet Confusion Matrix')
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
        
        colors = ['gold' if acc > 0.8 else 'darkgreen' if acc > 0.7 else 'green' if acc > 0.6 else 'lightgreen' if acc > 0.5 else 'orange' if acc > 0.3 else 'red' 
                 for acc in class_accuracies]
        
        bars = ax2.bar(range(len(self.class_names)), class_accuracies, color=colors)
        ax2.set_title('📊 Per-Class Accuracy (Hybrid MobileNet)')
        ax2.set_xlabel('Class')
        ax2.set_ylabel('Accuracy')
        ax2.set_xticks(range(len(self.class_names)))
        ax2.set_xticklabels(self.class_names, rotation=45)
        ax2.axhline(y=0.70, color='red', linestyle='--', alpha=0.7, label='Target (70%)')
        ax2.axhline(y=0.80, color='green', linestyle='--', alpha=0.7, label='Excellent (80%)')
        ax2.legend()
        
        # Add value labels
        for bar, acc in zip(bars, class_accuracies):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def plot_training_curves(self):
        """
        Plot training curves with stage indicators
        """
        if self.history is None:
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs = range(1, len(self.history.history['accuracy']) + 1)
        stage1_end = 30  # Where stage 1 ended
        
        # Accuracy
        ax1.plot(epochs, self.history.history['accuracy'], 'b-', label='Training', linewidth=2)
        ax1.plot(epochs, self.history.history['val_accuracy'], 'r-', label='Validation', linewidth=2)
        ax1.axvline(x=stage1_end, color='purple', linestyle='--', alpha=0.7, label='Stage 1→2')
        ax1.set_title('📈 Hybrid MobileNet Training Progress')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0.70, color='green', linestyle='--', alpha=0.7, label='Target')
        
        # Add stage labels
        ax1.text(15, 0.2, 'Stage 1:\nTrain Custom\nLayers', ha='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        ax1.text(55, 0.2, 'Stage 2:\nFine-tune\nHybrid Model', ha='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
        
        # Loss
        ax2.plot(epochs, self.history.history['loss'], 'b-', label='Training', linewidth=2)
        ax2.plot(epochs, self.history.history['val_loss'], 'r-', label='Validation', linewidth=2)
        ax2.axvline(x=stage1_end, color='purple', linestyle='--', alpha=0.7, label='Stage 1→2')
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
        
        print(f"\n📊 Hybrid Training Summary:")
        print(f"  Best validation accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
        print(f"  Best epoch: {best_epoch}")
        print(f"  Total epochs: {len(epochs)}")
        print(f"  Stage 1 (custom layers): 1-{stage1_end}")
        print(f"  Stage 2 (fine-tune hybrid): {stage1_end+1}-{len(epochs)}")

# Main execution
if __name__ == "__main__":
    print("🚀 HYBRID MOBILENET + CUSTOM CNN")
    print("="*60)
    
    trainer = HybridMobileNetCNN()
    
    try:
        print("TensorFlow version:", tf.__version__)
        print("Is GPU available:", tf.config.list_physical_devices('GPU'))

        final_accuracy = trainer.run_hybrid_training()
        
        print(f"\n🎉 HYBRID TRAINING COMPLETED!")
        print(f"🎯 Final accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
        
        if final_accuracy > 0.70:
            print(f"🏆 SUCCESS! Hybrid MobileNet is excellent!")
        elif final_accuracy > 0.65:
            print(f"🎉 VERY GOOD! Hybrid approach working well!")
        else:
            print(f"✅ Good progress with hybrid architecture!")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n✅ Hybrid MobileNet training completed!")
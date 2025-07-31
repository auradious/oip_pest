import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

class InsectCNNModel:
    def __init__(self, dataset_path="dataset", img_height=150, img_width=150, batch_size=32):
        """
        Initialize the Insect CNN Model
        
        Args:
            dataset_path: Path to the dataset directory
            img_height: Height of input images
            img_width: Width of input images
            batch_size: Batch size for training
        """
        self.dataset_path = dataset_path
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        self.model = None
        self.class_names = None
        self.history = None
        
    def load_and_preprocess_data(self):
        """
        Load and preprocess the dataset
        """
        print("Loading and preprocessing data...")
        
        # Create datasets
        train_ds = tf.keras.utils.image_dataset_from_directory(
            self.dataset_path,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size
        )
        
        val_ds = tf.keras.utils.image_dataset_from_directory(
            self.dataset_path,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size
        )
        
        # Get class names
        self.class_names = train_ds.class_names
        print(f"Found {len(self.class_names)} classes: {self.class_names}")
        
        # Configure for performance
        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
        
        return train_ds, val_ds
    
    def create_data_augmentation_layer(self):
        """
        Create data augmentation layer for better generalization
        """
        return tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
        ])
    
    def create_cnn_model(self):
        """
        Create CNN model following the specified architecture:
        Input Image → Convolution → Feature Maps → Pooling → Pooled Feature Maps → 
        Convolution → Feature Maps → Pooling → Pooled Maps → Flatten → CNN → Output
        """
        print("Creating CNN model with specified architecture...")
        
        # Data augmentation
        data_augmentation = self.create_data_augmentation_layer()
        
        # Model architecture
        model = tf.keras.Sequential([
            # Input layer
            layers.Input(shape=(self.img_height, self.img_width, 3)),
            
            # Data augmentation
            data_augmentation,
            
            # Rescaling layer (normalize pixel values to [0,1])
            layers.Rescaling(1./255),
            
            # First Convolution Block
            # Input Image → (Convolution) → Feature Maps
            layers.Conv2D(32, (3, 3), activation='relu', name='conv1'),
            layers.BatchNormalization(),
            
            # (Pooling) → Pooled Feature Maps
            layers.MaxPooling2D((2, 2), name='pool1'),
            
            # Second Convolution Block
            # (Convolution) → Feature Maps
            layers.Conv2D(64, (3, 3), activation='relu', name='conv2'),
            layers.BatchNormalization(),
            
            # (Pooling) → Pooled Maps
            layers.MaxPooling2D((2, 2), name='pool2'),
            
            # Third Convolution Block (Additional for better feature extraction)
            layers.Conv2D(128, (3, 3), activation='relu', name='conv3'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2), name='pool3'),
            
            # Fourth Convolution Block
            layers.Conv2D(256, (3, 3), activation='relu', name='conv4'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2), name='pool4'),
            
            # Flatten Layer
            # Pooled Maps → Flatten Layer
            layers.Flatten(name='flatten'),
            
            # CNN (Fully Connected) Layers
            layers.Dense(512, activation='relu', name='dense1'),
            layers.Dropout(0.5, name='dropout1'),
            
            layers.Dense(256, activation='relu', name='dense2'),
            layers.Dropout(0.3, name='dropout2'),
            
            # Output Layer
            layers.Dense(len(self.class_names), activation='softmax', name='output')
        ])
        
        self.model = model
        return model
    
    def compile_model(self, learning_rate=0.001):
        """
        Compile the model with optimizer, loss, and metrics
        """
        print("Compiling model...")
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=['accuracy']
        )
    
    def display_model_summary(self):
        """
        Display model architecture summary
        """
        print("\n" + "="*50)
        print("MODEL ARCHITECTURE SUMMARY")
        print("="*50)
        self.model.summary()
        
        # Display the flow as specified
        print("\n" + "="*50)
        print("PROCESSING FLOW:")
        print("="*50)
        print("Input Image → (Convolution) → Feature Maps → (Pooling) → Pooled Feature Maps")
        print("→ (Convolution) → Feature Maps → (Pooling) → Pooled Maps → Flatten Layer → CNN → Output")
        print("="*50)
    
    def train_model(self, train_ds, val_ds, epochs=25):
        """
        Train the model
        """
        print(f"Training model for {epochs} epochs...")
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3,
                min_lr=0.0001,
                verbose=1
            )
        ]
        
        # Train the model
        self.history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def plot_training_history(self):
        """
        Plot training and validation accuracy and loss
        """
        if self.history is None:
            print("No training history found. Train the model first.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def evaluate_model(self, val_ds):
        """
        Evaluate the model and display results
        """
        print("Evaluating model...")
        
        # Get predictions
        predictions = self.model.predict(val_ds)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Get true labels
        true_labels = []
        for images, labels in val_ds:
            true_labels.extend(labels.numpy())
        
        # Print classification report
        print("\nClassification Report:")
        print("="*50)
        print(classification_report(
            true_labels, 
            predicted_classes, 
            target_names=self.class_names
        ))
        
        # Create confusion matrix
        cm = confusion_matrix(true_labels, predicted_classes)
        
        # Plot confusion matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Calculate and display accuracy
        test_loss, test_accuracy = self.model.evaluate(val_ds, verbose=0)
        print(f"\nFinal Test Accuracy: {test_accuracy:.4f}")
        print(f"Final Test Loss: {test_loss:.4f}")
        
        return test_accuracy, test_loss
    
    def save_model(self, filepath="insect_cnn_model.h5"):
        """
        Save the trained model
        """
        if self.model is None:
            print("No model to save. Train the model first.")
            return
        
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath="insect_cnn_model.h5"):
        """
        Load a pre-trained model
        """
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
    
    def predict_single_image(self, image_path):
        """
        Predict the class of a single image
        """
        if self.model is None:
            print("No model loaded. Train or load a model first.")
            return
        
        # Load and preprocess image
        img = tf.keras.utils.load_img(
            image_path, 
            target_size=(self.img_height, self.img_width)
        )
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create batch
        
        # Make prediction
        predictions = self.model.predict(img_array)
        predicted_class = self.class_names[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))
        
        print(f"Predicted class: {predicted_class}")
        print(f"Confidence: {confidence:.4f}")
        
        return predicted_class, confidence


def main():
    """
    Main function to run the complete CNN training pipeline
    """
    print("="*60)
    print("INSECT CLASSIFICATION CNN MODEL")
    print("="*60)
    
    # Initialize the model
    cnn_model = InsectCNNModel(
        dataset_path="dataset",
        img_height=150,
        img_width=150,
        batch_size=32
    )
    
    # Load and preprocess data
    train_ds, val_ds = cnn_model.load_and_preprocess_data()
    
    # Create the CNN model
    model = cnn_model.create_cnn_model()
    
    # Compile the model
    cnn_model.compile_model(learning_rate=0.001)
    
    # Display model summary
    cnn_model.display_model_summary()
    
    # Train the model
    history = cnn_model.train_model(train_ds, val_ds, epochs=25)
    
    # Plot training history
    cnn_model.plot_training_history()
    
    # Evaluate the model
    accuracy, loss = cnn_model.evaluate_model(val_ds)
    
    # Save the model
    cnn_model.save_model("insect_cnn_model.h5")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print(f"Final Accuracy: {accuracy:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()

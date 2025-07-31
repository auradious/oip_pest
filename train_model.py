"""
Simple training script for the Insect CNN Model
Run this after setting up the environment
"""

def run_training():
    """Run the complete training pipeline"""
    print("Starting Insect CNN Model Training...")
    
    # Import the model
    from insect_cnn_model import InsectCNNModel
    
    # Initialize the model
    print("Initializing CNN model...")
    cnn_model = InsectCNNModel(
        dataset_path="dataset",
        img_height=150,
        img_width=150,
        batch_size=32
    )
    
    # Load and preprocess data
    print("Loading dataset...")
    train_ds, val_ds = cnn_model.load_and_preprocess_data()
    
    # Create the CNN model
    print("Creating CNN architecture...")
    model = cnn_model.create_cnn_model()
    
    # Compile the model
    print("Compiling model...")
    cnn_model.compile_model(learning_rate=0.001)
    
    # Display model summary
    cnn_model.display_model_summary()
    
    # Ask user if they want to proceed with training
    response = input("\nDo you want to start training? (y/n): ").lower().strip()
    if response != 'y':
        print("Training cancelled.")
        return
    
    # Train the model
    print("Starting training (this may take a while)...")
    history = cnn_model.train_model(train_ds, val_ds, epochs=25)
    
    # Plot training history
    print("Generating training plots...")
    cnn_model.plot_training_history()
    
    # Evaluate the model
    print("Evaluating model...")
    accuracy, loss = cnn_model.evaluate_model(val_ds)
    
    # Save the model
    print("Saving model...")
    cnn_model.save_model("insect_cnn_model.h5")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print(f"Final Accuracy: {accuracy:.4f}")
    print(f"Final Loss: {loss:.4f}")
    print("Model saved as: insect_cnn_model.h5")
    print("Training plots saved as: training_history.png")
    print("Confusion matrix saved as: confusion_matrix.png")
    print("="*60)

if __name__ == "__main__":
    try:
        run_training()
    except Exception as e:
        print(f"Error during training: {e}")
        print("Make sure you have:")
        print("1. Activated the virtual environment")
        print("2. Installed all required packages")
        print("3. Have the dataset folder with insect images")

"""
Pest Prediction Logic
Handles AI model loading and prediction processing
"""

import numpy as np
import json
import sys
from pathlib import Path
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config.config import *
from config.languages import LANGUAGES, DEFAULT_LANGUAGE

class PestPredictor:
    """
    Handles pest identification and treatment recommendations
    """
    
    def __init__(self):
        self.model = None
        self.class_names = {}
        self.load_model_and_classes()
        
    def load_model_and_classes(self):
        """
        Load the trained model and class mappings
        """
        try:
            # Try to load TensorFlow and model
            import tensorflow as tf
            
            # Load trained model
            model_path = MODEL_PATHS['best_model']
            if model_path.exists():
                self.model = tf.keras.models.load_model(model_path)
                print(f"✅ Model loaded from {model_path}")
            else:
                print(f"❌ Model not found at {model_path}")
                self.model = None
                
            # Load class mappings
            class_names_path = MODEL_PATHS['class_names']
            if class_names_path.exists():
                with open(class_names_path, 'r') as f:
                    self.class_names = json.load(f)
                print(f"✅ Class names loaded from {class_names_path}")
            else:
                print(f"❌ Class names not found at {class_names_path}")
                # Fallback to config classes
                self.class_names = {
                    'idx_to_class': {str(i): cls for i, cls in enumerate(HARMFUL_PEST_CLASSES)}
                }
                
        except ImportError:
            print("❌ TensorFlow not available - running in demo mode")
            self.model = None
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model = None
    
    def preprocess_image(self, image):
        """
        Preprocess uploaded image for model prediction
        """
        if image is None:
            return None
            
        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        # Resize to model input size
        target_size = IMAGE_CONFIG['target_size']
        image = image.resize(target_size)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array and normalize
        img_array = np.array(image)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = img_array.astype('float32') / 255.0  # Normalize
        
        return img_array
    
    def get_treatment_recommendation(self, pest_class, language='en'):
        """
        Get organic treatment recommendations for identified pest
        """
        lang_data = LANGUAGES.get(language, LANGUAGES[DEFAULT_LANGUAGE])
        
        if pest_class not in HARMFUL_PEST_CLASSES:
            return lang_data['predictions']['no_treatment']
        
        # Get pest characteristics
        economic_impact = ECONOMIC_IMPACT.get(pest_class, 3)
        urgency = TREATMENT_URGENCY.get(pest_class, 'Medium')
        
        # Get treatment text for the pest
        treatment_text = lang_data['treatments'].get(pest_class, '')
        organic_treatments = lang_data['treatments']['organic_treatments']
        
        treatment = f"{organic_treatments}\n{treatment_text}"
        
        # Add urgency and impact information
        impact_emojis = {1: "💚", 2: "💛", 3: "🧡", 4: "❤️", 5: "💔"}
        urgency_emojis = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
        
        # Get localized urgency text
        urgency_level = lang_data['predictions']['urgency_level']
        economic_impact_text = lang_data['predictions']['economic_impact']
        action_priority = lang_data['predictions']['action_priority']
        
        # Get action priority text based on urgency
        if urgency == 'High':
            priority_text = lang_data['predictions']['immediate_action']
        elif urgency == 'Medium':
            priority_text = lang_data['predictions']['action_recommended']
        else:
            priority_text = lang_data['predictions']['monitor_situation']
        
        recommendation = f"""**{urgency_emojis[urgency]} {urgency_level} {urgency}**
**{impact_emojis[economic_impact]} {economic_impact_text} {economic_impact}/5**

{treatment}

**{action_priority}**
{priority_text}"""
        
        return recommendation
    
    def predict_pest(self, image, language='en'):
        """
        Main prediction function called by the interface
        """
        lang_data = LANGUAGES.get(language, LANGUAGES[DEFAULT_LANGUAGE])
        
        if image is None:
            return lang_data['predictions']['no_image'], lang_data['predictions']['no_image_desc']
        
        if self.model is None:
            return (f"{lang_data['predictions']['model_not_loaded']}\n\nPlease train the model first using:\n"
                   "1. `python src/data_preprocessing.py`\n"
                   "2. `python src/model_training.py`"), lang_data['predictions']['model_not_loaded_desc']
        
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image)
            
            # Make prediction
            predictions = self.model.predict(processed_image, verbose=0)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])
            
            # Get class name
            predicted_class = self.class_names['idx_to_class'].get(str(predicted_class_idx), 'Unknown')
            
            # Format prediction result
            confidence_percent = confidence * 100
            
            if confidence < 0.3:
                result = f"{lang_data['predictions']['uncertain']}\n\n{lang_data['predictions']['most_likely']} {predicted_class.title()}\n{lang_data['predictions']['confidence']} {confidence_percent:.1f}%\n\n{lang_data['predictions']['uncertain_desc']}"
                recommendation = lang_data['predictions']['recommendation_unclear']
            else:
                result = f"{lang_data['predictions']['identified']}\n\n{lang_data['predictions']['species']} {predicted_class.title()}\n{lang_data['predictions']['confidence']} {confidence_percent:.1f}%"
                recommendation = self.get_treatment_recommendation(predicted_class, language)
            
            return result, recommendation
            
        except Exception as e:
            return f"{lang_data['predictions']['error']} {str(e)}", lang_data['predictions']['error_desc']
